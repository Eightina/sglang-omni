# MiniCPM-o 4.5 本地部署指南（sglang-omni）

> 本文档记录在本机（单卡 RTX 5090 32GB，CUDA 13.2 驱动，Ubuntu 24.04）复用现有环境
> 部署 sglang-omni + MiniCPM-o 4.5 的完整流程，包括踩过的坑。下次部署直接照此执行。

## 0. 前提条件核对

| 项目 | 要求 | 本机实际情况 |
|---|---|---|
| GPU | ≥24GB 显存（bf16 权重约 18GB） | RTX 5090 32GB |
| 驱动 | CUDA 13.x（匹配 flash-attn-4/nixl cu13 轮子） | 595.58.03 / CUDA 13.2 |
| Python | 3.10–3.12 | 3.12.3 |
| 权重 | MiniCPM-o 4.5 完整目录（含 4 个 safetensors 分片 + 自定义 py 文件） | `/cpm-workspace/models/MiniCPM-o-4_5`（bf16，约 18GB） |
| 磁盘 | 全新安装需 8–10GB；复用环境只需 <1GB | 约 11GB 可用（紧张，故走复用路线） |
| 网络 | 需能访问 GitHub / PyPI | SSH 首次克隆需先 `ssh-keyscan github.com >> ~/.ssh/known_hosts` |

**权重完整性检查**：

```bash
ls /cpm-workspace/models/MiniCPM-o-4_5/
# 必须有: model-0000X-of-00004.safetensors × 4、config.json(torch_dtype=bfloat16)、
# modeling_minicpmo.py、configuration_minicpmo.py、processing_minicpmo.py、
# tokenizer.json、model.safetensors.index.json
```

## 1. 拉取仓库

```bash
mkdir -p ~/.ssh && ssh-keyscan -t ed25519 github.com >> ~/.ssh/known_hosts 2>/dev/null
cd /sglomni-workspace
git clone git@github.com:sgl-project/sglang-omni.git
# 验证版本：MiniCPM-o 支持于 2026-09-18 引入（commit 524a3843，PR #1879），
# 仅存在于 v0.1.7 及之后的 main 分支；v0.1.6 及更早不含该模型
git log --oneline -1
```

## 2. 环境策略：复用 /venv/main（关键决策）

`/venv/main` 是 `/sgl-workspace` 那套 SGLang 0.5.16 项目的运行环境（Python 3.12 venv，
`--system-site-packages`，大部分包在系统 dist-packages），由 `/sgl-workspace/constraints.txt`
锁定版本。它已覆盖 sglang-omni 绝大部分重依赖（torch/flash-attn-4/flashinfer/kernels/msgspec
等 44 项），只有少量小包缺失。

**为什么不能直接用老栈跑 MiniCPM-o**：sglang-omni（含 MiniCPM-o 的所有版本）框架层深度
依赖 sglang ≥0.5.19/0.5.20 的新 API（`sglang.srt.arg_groups.model_override_base` 模块、
`runtime_context` 的 `get_schedule/get_exec/publish` 等），sglang 0.5.16 无法用 shim 弥补。

**为什么不全新安装**：官方 pin（torch==2.13.0 + CUDA 13 全家桶）需下载 8–10GB，磁盘紧张。
实测在 torch 2.11.0 上运行 sglang 0.5.20 + sglang-omni 完全可行（见第 7 节偏差清单）。

**快速盘点环境**（部署前跑一遍，确认缺口）：

```bash
/venv/main/bin/python -c "
import importlib
need = ['pyzmq','msgpack','msgspec','pydantic','yaml','pybase64','torch','torchvision',
'accelerate','transformers','safetensors','PIL','huggingface_hub','fastapi','uvicorn',
'cache_dit','addict','imageio','httpx','xxhash','sox','av','qwen_vl_utils','numba','librosa',
'sentencepiece','pandas','tabulate','typer','openai','openai_harmony','soundfile',
'mistral_common','silero_vad','onnxruntime','websockets','scipy','whisper','tiktoken',
'omegaconf','torchaudio','torchcodec','gradio','einops','onnx','kernels','diffusers','sglang']
miss = [m for m in need if not __import__('importlib').util.find_spec(m)]
print('MISSING:', miss)
"
```

## 3. 安装步骤（复用环境路线）

```bash
# 3.1 editable --no-deps 安装 sglang-omni 本体（注册 sgl-omni CLI 和 serve backend 插件）
/venv/main/bin/pip install --no-deps -e /sglomni-workspace/sglang-omni

# 3.2 补齐缺失小包（constraints 锁底，防止已装的 torch/numba/transformers 等被升级）
/venv/main/bin/pip install -c /sgl-workspace/constraints.txt \
    pyzmq accelerate librosa onnx sox qwen-vl-utils openai-whisper
/venv/main/bin/pip install -c /sgl-workspace/constraints.txt onnxruntime-gpu   # minicpm_o.stages 需要
/venv/main/bin/pip install -c /sgl-workspace/constraints.txt silero-vad        # API server 启动硬依赖

# 3.3 升级 sglang 到 0.5.20（--no-deps 只换纯 Python 代码，torch 2.11 和 sglang-kernel 0.4.5 不动）
/venv/main/bin/pip install --no-deps --upgrade sglang==0.5.20

# 3.4 导入自检（应全部 OK）
/venv/main/bin/python -c "
import sglang; print('sglang', sglang.__version__)
import sglang_omni, sglang_omni.models.minicpm_o.stages
from sglang.srt.arg_groups.model_override_base import resolved_view
from sglang.srt.runtime_context import get_schedule
print('imports OK')
" 2>&1 | grep -v "Ignore import\|torchada"
```

可跳过的依赖（评测/playground 用，体积大）：`gradio`、`s3prl`、`nemo_text_processing`、
`jiwer`、`zhon`、`hydra-core`、`x-transformers`、`dots.tts`。用到对应功能时再补装。

## 4. 启动服务

```bash
cd /sglomni-workspace
HF_HUB_OFFLINE=1 /venv/main/bin/sgl-omni serve \
    --model-path /cpm-workspace/models/MiniCPM-o-4_5 \
    --host 0.0.0.0 --port 30000 \
    --thinker.engine.mem_fraction_static 0.80 \
    --talker.engine.mem_fraction_static 0.10 \
    < /dev/null 2>&1 | tee /tmp/sglomni_serve.log
```

### 4.1 两个必踩的坑

**坑 1：KV cache 显存不足（不加显存参数 100% 复现）**

默认自动计算 `mem_fraction_static=0.57`，thinker 权重（17.4GB）装载后剩余空间不足，
报错退出：

```
ValueError: Loaded weights leave no GPU memory for the KV cache under
--mem-fraction-static=0.57. Raise --mem-fraction-static above 0.788 ...
```

解决：显式分 stage 指定（thinker 与 talker 同卡共享 32GB，不能都用大值）：

- `--thinker.engine.mem_fraction_static 0.80` → 25.6GB 预算 = 17.4GB 权重 + 约 7GB KV
- `--talker.engine.mem_fraction_static 0.10` → 3.2GB（talker 权重小，够用）

**坑 2：trust_remote_code 交互确认卡住启动**

MiniCPM-o 权重目录含自定义代码（`modeling_minicpmo.py`），transformers 5.x 会在启动时
弹出 `Do you wish to run the custom code? [y/N]` 交互确认并阻塞。本地部署无需真实应答：

- 命令末尾加 `< /dev/null`（EOF 走默认路径），架构解析有 raw-JSON 兜底
  （`try_resolve_arch_from_raw_config`），服务能正常拉起
- `HF_HUB_OFFLINE=1` 避免启动时访问 HuggingFace 网络

### 4.2 启动时长预期

首次启动约 **5–8 分钟**：权重装载 + torch inductor 为 talker/audio encoder 自动调优
编译 kernel + thinker 捕获 12 档 CUDA graph。日志出现
`Uvicorn running on http://0.0.0.0:30000` 即就绪。编译缓存在
`/root/.cache/torch_extensions/py312_cu130`，**保留它可显著加快下次启动**。

日志中的以下内容均为**无害噪音**，不要误判为失败：

- `Failed to import torchada` — MUSA 平台兼容探测，忽略
- `Ignore import error when loading sglang_omni.models.<其他模型>` — 注册表按需跳过无关模型
- `No valid triton configs. OutOfMemoryError: out of resource: triton_mm` — inductor
  自动调优中超出共享内存的候选配置被忽略，自动选用合法配置
- `Unable to import torchao Tensor objects` — torchao 未装，不影响 bf16 权重

## 5. 验证

```bash
# 5.1 健康检查（7 个 stage 全部就绪）
curl -s http://localhost:30000/health
# 期望: {"status":"healthy","stages":["preprocessing","thinker","image_encoder",
#        "audio_encoder","decode","talker","code2wav"],...}

# 5.2 文本对话
curl -s http://localhost:30000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "/cpm-workspace/models/MiniCPM-o-4_5",
  "messages": [{"role": "user", "content": "用一句话介绍你自己。"}],
  "max_tokens": 128, "modalities": ["text"]}'

# 5.3 语音输出（talker + code2wav 全链路；data 字段为 base64 WAV）
curl -s http://localhost:30000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "/cpm-workspace/models/MiniCPM-o-4_5",
  "messages": [{"role": "user", "content": "请用中文说：你好，很高兴认识你。"}],
  "modalities": ["text", "audio"], "audio": {"format": "wav"}}' | python3 -c "
import json,sys,base64
m = json.load(sys.stdin)['choices'][0]['message']
open('/tmp/out.wav','wb').write(base64.b64decode(m['audio']['data']))
print(m['content'])"

# 5.4 视觉理解（本地图片路径或 HTTP URL 均可）
curl -s http://localhost:30000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "/cpm-workspace/models/MiniCPM-o-4_5",
  "messages": [{"role": "user", "content": "这张图片里有什么？"}],
  "images": ["/cpm-workspace/models/MiniCPM-o-4_5/assets/fossil.png"],
  "modalities": ["text"], "max_tokens": 128}'

# 5.5 语音克隆（可选：传参考音频控制音色，参考 docs/cookbook/minicpm_o.md）
# audio.ref_audio 需为 base64 data URI；不传则用 checkpoint 自带 assets/HT_ref_audio.wav
```

注意：模型自称「Qwen」是**正常现象**——MiniCPM-o 的语言主干即 Qwen2.5-7B。

## 6. 故障排查速查

| 症状 | 原因 | 处理 |
|---|---|---|
| 启动卡在 `Do you wish to run the custom code?` | transformers 5.x 对自定义代码的交互确认 | `< /dev/null` 重定向 stdin |
| `Loaded weights leave no GPU memory for the KV cache ... mem-fraction-static=0.57` | 默认显存预算不适用于 18GB 权重 | 按第 4.1 节显式指定 thinker/talker 的 mem_fraction_static |
| `ModuleNotFoundError: No module named 'silero_vad'`（rich traceback） | API server 启动硬依赖 | `pip install -c /sgl-workspace/constraints.txt silero-vad` |
| `ModuleNotFoundError: No module named 'onnxruntime'`（import stages 时） | minicpm_o 的 speech tokenizer 依赖 | `pip install -c /sgl-workspace/constraints.txt onnxruntime-gpu` |
| `No module named 'sglang.srt.arg_groups.model_override_base'` | sglang 仍是 0.5.16 | 按第 3.3 步升级 sglang==0.5.20 |
| 端口未监听且日志无报错 | 还在 inductor 编译（首启 5–8 分钟） | `tail -f /tmp/sglomni_serve.log` 观察，勿提前杀进程 |
| 磁盘写满 | 安装了大依赖或缓存堆积 | 清 `/root/.cache/huggingface`（877MB）、pip cache；勿动 torch_extensions |

## 7. 版本偏差清单（复用环境的已知代价，实测可跑）

| 包 | sglang-omni 官方 pin | 本环境实际 | 影响 |
|---|---|---|---|
| torch | 2.13.0 | 2.11.0+cu130 | 无，推理链路全部通过 |
| torchvision | 0.28.0 | 0.26.0+cu130 | 无 |
| torchcodec | 0.15.0 | 0.11.1+cu130 | 视频输入未验证；文本/图片/音频链路正常 |
| flashinfer | 0.6.18 | 0.6.14 | 无 |
| transformers | 5.12.1 | 5.12.1（一致） | — |
| sglang-kernel | — | 0.4.5 | 无 |
| pillow | — | 被 constraints 降到 11.3.0 | 无 |

## 8. 回滚

```bash
# 恢复 /sgl-workspace 的 sglang 0.5.16 editable 安装（checkout 文件从未被改动）
/venv/main/bin/pip install --no-deps -e /sgl-workspace/sglang/python

# 卸载 sglang-omni
/venv/main/bin/pip uninstall sglang-omni
```

## 9. 附：全新机器的官方安装路线（不受磁盘/旧环境约束时）

```bash
git clone git@github.com:sgl-project/sglang-omni.git && cd sglang-omni
uv venv .venv -p 3.12 && source .venv/bin/activate
uv pip install --prerelease=allow -e ".[minicpm-o]"   # torch 2.13 + sglang 0.5.20 严格匹配
# 启动命令同第 4 节（mem_fraction_static 参数同样必带）
```

或使用 Docker（官方推荐，UCX/flash-attn/SGLang 预编译）：见
`docs/get_started/installation.md`。

---

*文档生成于 2026-10-02，基于 sglang-omni main（commit 1f089416）、sglang 0.5.20、
torch 2.11.0+cu130、MiniCPM-o 4.5（/cpm-workspace/models/MiniCPM-o-4_5）实测。*
