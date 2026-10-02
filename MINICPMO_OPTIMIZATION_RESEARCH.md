# MiniCPM-o 4.5 推理性能优化调研（Issue #2284 · 三项）

> 本文档针对 [sgl-project/sglang-omni#2284](https://github.com/sgl-project/sglang-omni/issues/2284)
> 中与 MiniCPM-o 4.5 语音链路直接相关的三项优化方向做代码级调研，逐项给出：
> **需求内容 / 现有基础 / 实现重难点与风险 / 工作量**。
>
> 调研基于本机已跑通的部署（单卡 RTX 5090 32GB，thinker/talker/code2wav 三进程同卡），
> 详见 `DEPLOYMENT_MINICPMO_LOCAL.md`。所有结论均带 `文件:行号` 证据，行号基于调研时的
> main 分支快照，改动前请以实际代码为准。

调研的三项需求：

1. **Thinker**：profile prefill/decode 调度、hidden-state 捕获与 per-token 拷贝；在实测基础上调优 SGLang 现有的 batching / graph / async-decode 能力。
2. **Thinker → Talker**：减少 hidden-state 序列化与 CPU/GPU 往返；参考 Qwen3-Omni 的流式 handoff，调研分级（staged）conditioning 传输。
3. **Flow / DiT**：调研 device-only CUDA graph 边界、重复的 per-step 准备、fused projections 与变长执行；对照 Qwen3-Omni graph runner 与 CosyVoice packed DiT。

---

## 0. 架构前提（决定三项难度的根因）

MiniCPM-o 语音链路目前是「**一次性批处理**」范式：

```
thinker 完整跑完 → 整段 hidden states 拷回 CPU → 序列化跨进程传给 talker
→ talker 一次性构建 condition → talker 出 codec → code2wav 一次性出音频
```

而 Qwen3-Omni 是「**流式 handoff**」范式（thinker 边生成边把 chunk 流给 talker，
talker 提前起、逐步 decode 门控）。三项需求本质上都是把 MiniCPM-o 从前者往后者迁移。

**关键有利条件**：仓库内已有三套可直接借鉴的成熟参考实现——
`qwen3_omni`（流式 handoff + code2wav CUDA graph）、`fun_cosyvoice3`（CosyVoice packed DiT + flow CUDA graph/compile/TRT）、
`auk`（flow-matching Euler step 的 device-only CUDA graph + fused qk-norm-rope）。
这意味着三项都**不需要从零发明轮子**，主要是「移植 + 适配 + 数值验证」。

pipeline stage 与进程放置（`sglang_omni/models/minicpm_o/config.py:147-159`）：

| stage | 进程 | GPU | 说明 |
|---|---|---|---|
| preprocessing / thinker / image_encoder / audio_encoder / decode | `pipeline` | 0 | thinker 复用 SGLang 原生 scheduler |
| talker | `talker` | 0 | **独立进程**，与 thinker 同卡 |
| code2wav | `code2wav` | 0 | **独立进程**，flow/DiT/HiFT 在此 |

> thinker→talker、talker→code2wav 都是**同卡跨进程**，必然涉及 IPC 与序列化——这是需求 2 的核心矛盾点。

---

## 1. 需求一 · Thinker（prefill/decode 调度、hidden-state 捕获、per-token 拷贝）

### 1.1 需求内容
Thinker 是语言主干（Qwen2.5/Qwen3-7B）。需要 profile 其 prefill/decode 调度，弄清
「为送 talker 而捕获 hidden state」这一步的开销（per-token 拷贝、同步点），并在实测基础上
调优 SGLang 已有的 continuous batching / CUDA graph / async-decode（overlap）能力。

### 1.2 现有基础（已复用的能力）
Thinker **并非自研前向循环，而是复用 SGLang 原生引擎**：

- 模型直接包 SGLang 的 `Qwen3ForCausalLM`：`components/sglang_thinker.py:39`。
- 引擎已开启多项优化（`stages.py:279-339` `create_sglang_thinker_executor_from_config`）：
  - `disable_cuda_graph=False`（:301）——CUDA graph 开启；
  - `enable_mixed_chunk=True`（:302）、`chunked_prefill_size=8192`（:303）——分块/混合 prefill；
  - `enable_async_decode=True`（:290/:331）、`async_decode_min_batch_size=2`（:291）——overlap decode；
  - `max_running_requests=64`（:299）。
- 共享基类 `model_runner/thinker_model_runner.py` 已提供成熟的 overlap 机制：
  `async_host_buf`（pinned 双缓冲，:542-555）、`post_decode_launch` / `post_decode_resolve`（:568-607）。

### 1.3 真正的瓶颈（都在「语音路径」上，且都源于 hidden 捕获约束）

1. **per-token GPU 拷贝**：`thinker_model_runner.py:99-102` 的 `post_process_outputs` 每个 decode step 都
   `hidden.detach().clone()`，注释明说「CUDA graph replay 会覆盖原 hidden buffer」，故必须逐个 clone，
   全部驻留 GPU 直到请求结束（显存压力 + 每步一次小 kernel/分配）。
2. **末尾阻塞式 D2H**：`thinker_model_runner.py:112-124` 的 `on_request_finished` 用
   `torch.stack(seq).to("cpu")`（:123）一次性同步拷回，**非 pinned、非 non_blocking**。
3. **语音请求被排除出 async decode（overlap）**：基类 `lookahead_eligible`
   （`model_runner/thinker_model_runner.py:495-540`）在 `should_generate_audio_output(...)` 时直接
   `return False`（:515），注释解释「MiniCPM-o 语音在 resolve 时才读 hidden，此时下一次 launch 已覆盖它」。
   → **语音 decode 是同步的，没有 plan/execute 重叠**，`async_host_buf` 那套 overlap 用不上。
4. **hidden 捕获与 CUDA graph 的张力**：基类默认 `requested_capture_hidden_mode_*` 返回 `NULL`
   （`model_runner/thinker_model_runner.py:84-100`），注释指出「请求 hidden 会把 batch capture mode 抬到
   graph 之上而跑 eager」。MiniCPM-o 覆写为 `FULL`（`thinker_model_runner.py:58-75`）——
   **需实测确认 graph 是否真能在 FULL 模式 replay，还是退化为 eager**。

### 1.4 实现重难点与风险
- **难点**：要让语音路径吃上 overlap，必须解决「resolve 读 hidden 时被下一次 launch 覆盖」的竞态
  （当前正是因此主动关闭 lookahead）。需要把 hidden 写入预分配的 pinned ring buffer，让捕获与读取解耦。
- **风险（低—中）**：
  - 调参类（batch、`async_decode_min_batch_size`、`chunked_prefill_size`）风险低，但收益需实测；
  - 改 hidden 捕获涉及**数值一致性**（clone→ring buffer 不能改变送 talker 的向量）；
  - 开语音 overlap 会与同卡 talker **抢 GPU/SM**，可能得不偿失，需在目标并发下度量。

### 1.5 工作量
| 子任务 | 估时 | 备注 |
|---|---|---|
| Profiling（分段耗时基线） | **2–4 天** | 先做；是另两项的度量底座。可用 `.claude/skills/model-profiling`、`omni-gpu-deep-dive` |
| 低风险参数调优 | **2–3 天** | 基于 profile 结果 |
| 重写 hidden 捕获（消除 per-token clone / 末尾阻塞 D2H → pinned 增量拷贝） | **1–2 周** | 含数值一致性验证 |
| 让语音 decode 吃上 async overlap | **+1–2 周** | 竞态 + 抢卡取舍，可选 |

**综合难度：中。**

---

## 2. 需求二 · Thinker → Talker（减少序列化与 CPU/GPU 往返，分级 conditioning 传输）

### 2.1 需求内容
Thinker 产出的 hidden states（conditioning 向量）要送到独立进程的 talker。当前存在完整的
**GPU→CPU→序列化→IPC→CPU→GPU** 往返，且是**一次性**传输。目标是减少序列化与往返，
并参考 Qwen3-Omni 的流式 handoff 实现「分级/分阶段 conditioning 传输」。

### 2.2 现有基础（当前数据通路）
1. thinker 末尾 `.to("cpu")` + `unbind` 成一列 CPU tensor：`thinker_model_runner.py:112-124`。
2. `project_thinker_to_talker` 把整段 `hidden_states_seq` 塞进 payload，`to_dict()` 后跨进程传输：
   `routing.py:172-186`、`routing.py:239-246`。
3. 传输层 `comm/stage_io.py`：`extract_tensors`（占位符 + `pickle`，:8/:89-119）走 CPU 通道。
4. talker 侧 `build_talker_request` 再 `torch.stack(...)`（CPU，`talker_request.py:79`），
   `build_condition_embeddings` 里 `.to(device)`（H2D）+ projector + normalize：`components/sglang_talker.py:93-125`。

结构事实：thinker 只 `stream_to=["decode"]`（文本，`config.py:76`），**没有 stream 到 talker**；
talker 请求由一次性 payload 构建（`config.py:98-109`）。

### 2.3 已有但未用上的能力（关键机会）
仓库**已有 GPU 零拷贝跨进程通道**：
- `comm/stage_io.py:122-156` `extract_cuda_tensors` + `DirectCudaIpcPayloadRef`/`DirectCudaIpcStreamChunkRef`（:26-30）；
- `relay/cuda_ipc.py`（1450 行）——同卡跨进程 CUDA IPC 直传 GPU tensor；
- `relay/` 另有 `nccl.py`、`shm.py` 等后端。

MiniCPM-o 因为先 `.to("cpu")`，这条边**用不上 CUDA IPC**，被迫走 CPU 序列化路径。
（对照：`config.py:54` 的 audio_encoder 显式 `disable_direct_cuda_ipc_payload=True`，说明该开关是 per-edge 可控的。）

### 2.4 Qwen3-Omni 流式 handoff 参考
- `talker_scheduler.py:50-123` `QwenTalkerScheduler`：`enable_partial_start` + `is_request_build_ready`，
  攒够 `TALKER_START_MIN_CHUNKS` 个 chunk 就**提前起 talker**；
- 逐步 decode 门控 `is_batch_ready_to_run` / `note_chunk_wait` / `rollback_decode_prep_after_skip`
  （`talker_scheduler.py:137-192`）：chunk 没到就推迟并回滚 decode 预分配 → **thinker/talker 真正重叠**；
- talker 请求从 `payload.prefetched_chunks` 流式构建（`request_builders.py:1284-1285`），
  conditioning 更丰富（dual-layer `talker_layer_hidden_states` + `multimodal_mask`，`request_builders.py:998-1003`），
  并用 `future_text_rows` FIFO 流增量喂入（`components/talker_input.py:150-161`）。这就是 issue 说的 staged conditioning transfer。

### 2.5 实现重难点与风险
- **子方向 (a) 让 hidden 常驻 GPU 走 CUDA IPC**——难度**中**：机制现成，主要是把这条边接上、
  管理跨进程 IPC 生命周期/CUDA event、让 router 放行 CUDA tensor（`comm/router.py` 有 `tensor_devices` 校验）。
- **子方向 (b) 改造成流式 handoff**——难度**高**：架构级改动。MiniCPM-o 的 talker 现在依赖「整段序列」
  来切 `tts_bos..tts_eos` span 并一次性算 condition（`talker_request.py:26-80`），改成增量 chunk +
  partial start + 逐步门控，需动 thinker runner、routing、talker scheduler、request builder。
- **风险**：
  - (a) 中风险——IPC 生命周期/event 同步出错会导致读到未写完的 buffer（数据竞争）；
  - (b) **高风险**——直接触及**音质与时序**，流式切分 span 的边界（`tts_bos/eos`、history turn）极易出错，
    必须做端到端音质与尾延迟验证；
  - 两方向都需保证与一次性路径的**数值等价**。

### 2.6 工作量
| 子任务 | 估时 | 备注 |
|---|---|---|
| (a) hidden 常驻 GPU + CUDA IPC 直传 | **1–2 周** | 高性价比，建议先做 |
| 「分级传输」折中版（分若干 bounded stage 传，而非逐 token / 整段） | **+3–5 天** | 在 (a) 之上 |
| (b) 完整流式 handoff（partial start + 逐步门控） | **3–6 周** | 含大量音质/尾延迟验证 |

**综合难度：高（三项里最深）。** 建议 (a) 先落地拿收益，(b) 视 (a) 效果与音质验证再排期。

---

## 3. 需求三 · Flow / DiT（device-only CUDA graph、per-step 准备、fused projection、变长执行）

### 3.1 需求内容
code2wav 里的 flow-matching DiT 把 codec token 转成 mel，再由 HiFT 出波形。需要调研：
device-only CUDA graph 边界、每步重复的准备计算、fused projections、变长执行；对照
Qwen3-Omni graph runner 与 CosyVoice packed DiT。

### 3.2 现有基础（现状 = 10 步 eager，无 CUDA graph）
- `components/token2wav/flow.py:46-91` `solve_euler`：**10 步 eager Euler 循环**，每步 `torch.cat([x,x])`（CFG 加倍，:68-69）。
- `flow.py:153-225` `CausalMaskedDiffWithXvec.inference`：每次 vocode 有 `.tolist()` 同步（:170-171）、
  Python 循环 `pad_sequence`（:172-182）、逐行 scatter `mel_conditioning`（:200-202）。
- `components/token2wav/dit.py`：
  - **已有 varlen packed 路径** `forward_packed`（:107-128、:467-508，用 `varlen_attn`）；
  - 但 `sequence_ids / cumulative_sequence_lengths / real_frame_positions / real_frame_mask`
    **每步都重算**（:474-494），而这些在 10 步内不变 → 典型「repeated per-step preparation」，可提到循环外；
  - `Attention` 的 `to_q/to_k/to_v` 三个独立 Linear（:69-71）→ 可 fused QKV；
  - 变长门控 `enable_variable_length and batch>=3 and is_cuda`（:450-454）。
- **变长批处理已有雏形**：`code2wav.py:263-325` `vocode` 已按 token 长度分桶跑 HiFT（:309-320），
  末尾单次 `.cpu()`（:321-323）；executor 默认 `enable_flow_variable_length=True`（`stages.py:209-261`）。

### 3.3 仓库内现成参考（本项最大优势）
- **auk（最贴合的模板）**：`models/auk/step_cuda_graph.py:135-152` `AuKStepCudaGraphRunner`——
  「每个声明 shape 捕获**一个 Euler step**，对每个 NFE step replay」，配 `CapturedStep` 静态缓冲、
  `verify_capture_shapes`、`DEFAULT_CAPTURE_SHAPES`、`warmup_iters`；fused 参考 `models/auk/fused_qk_norm_rope.py`。
- **fun_cosyvoice3（CosyVoice packed DiT）**：`config.py:19-75` 列出 **54 个 (batch,length) 捕获 shape**；
  `config.py:162-167` `enable_flow_cuda_graph` / `enable_dit_torch_compile` / `enable_flow_estimator_trt`；
  `packed_dit.py:192-248` `PackedDiT` + `RaggedRowAttention`（FA3，no pad-to-widest）；`flow_estimator_trt.py`（567 行）。
- **qwen3_omni**：`components/code2wav_cuda_graph.py:76-244` `Code2WavCudaGraphRunner` / `GraphKey` /
  `CapturedGraph` / `graph_pool_handle`——带 graph pool、按 shape keyed 捕获、stats 与失败兜底。

### 3.4 实现重难点与风险
- **难点**：CUDA graph 要求**静态 shape** → 需分桶 + padding（CosyVoice 的 54-shape 列表即范例），
  并与现有变长 `forward_packed` 路径协同；CFG 加倍、autocast/bf16 边界要固定；捕获 shape 太多会吃显存。
- **风险（中）**：issue 明确要求 **checkpoint 级数值等价 + final-head GPU 验证**。CUDA graph/compile/TRT
  都可能引入微小数值差，需逐档 shape 校验；分桶 padding 会改变 HiFT 非因果卷积边界（`code2wav.py:313` 已有注释）。
- **有利点**：MiniCPM-o 的 DiT 与 CosyVoice 同源（`flow.py` 头即 Alibaba 版权），移植阻力小。

### 3.5 工作量
| 子任务 | 估时 | 备注 |
|---|---|---|
| per-step 不变量提到循环外（`forward_packed` 的 sequence_ids/cumsum/mask） | **2–4 天** | 低风险，先做 |
| fused QKV projection | **3–5 天** | 参考 auk `fused_qk_norm_rope.py` |
| Euler step 的 CUDA graph 捕获 + shape 分桶 | **1.5–3 周** | 照搬 auk `AuKStepCudaGraphRunner` 模板，含数值等价验证 |
| （可选）torch.compile / TRT estimator | **+1–2 周** | 非必需，参考 CosyVoice `flow_estimator_trt.py` |

**综合难度：中（模板最全、ROI 最高）。**

---

## 4. 汇总对照与推进建议

### 4.1 对照表
| 需求 | 重点 | 难度 | 工作量 | 现成参考 | 收益/风险 |
|---|---|---|---|---|---|
| **1 Thinker** | 语音路径 per-token clone + 末尾阻塞 D2H + overlap 被关；先 profile 再定点调 | 中 | profile+调优 2–4 天；重写捕获 1–2 周；语音 overlap +1–2 周 | 基类 `async_host_buf`/`lookahead` overlap | 中收益 / 低—中风险 |
| **2 Thinker→Talker** | 消除 GPU→CPU→序列化→CPU→GPU 往返；一次性→流式/分级 | 高 | (a) GPU-IPC 直传 1–2 周；(b) 完整流式 3–6 周 | Qwen3 `QwenTalkerScheduler`；`relay/cuda_ipc.py` | (a) 高收益/中风险；(b) 高收益/高音质风险 |
| **3 Flow/DiT** | 10 步 eager→CUDA graph；per-step 外提；fused QKV；变长分桶 | 中 | 外提 2–4 天；fused 3–5 天；CUDA graph 1.5–3 周 | auk `AuKStepCudaGraphRunner`、CosyVoice 捕获 shapes、Qwen3 `Code2WavCudaGraphRunner` | **高收益 / 中风险，模板最全** |

### 4.2 建议推进顺序
1. **先做需求 1 的 profiling（2–4 天）**：issue 反复强调「measured / numerical validation before changing
   defaults」，这也是判断另两项收益的度量底座。用 `.claude/skills/model-profiling`、`omni-gpu-deep-dive`
   采 prefill / decode / handoff / vocode 的分段耗时。
2. **需求 3 优先落地**（尤其 per-step 外提 + Euler step CUDA graph）：最自包含、仓库内模板最全（auk 近乎
   drop-in）、不触碰音质敏感的跨 stage 语义，ROI 最高。
3. **需求 2 先做 (a) GPU-IPC 直传**：拿掉最大的 CPU/GPU 往返且改动可控；完整流式 handoff (b) 作为独立
   较大工程，视 (a) 收益与音质验证再排期。

### 4.3 通用验收红线（三项共用）
- **数值等价**：改动前后送 talker 的 hidden、flow 输出的 mel、最终波形需逐档校验；
- **音质**：语音输出主观/客观指标不回退（尤其需求 2(b)、需求 3 的分桶 padding）；
- **尾延迟 & 显存**：在目标并发下度量，CUDA graph 捕获 shape 与 hidden 常驻 GPU 都会增显存；
- **默认值保守**：先以可选开关落地，实测确认后再改默认。

---

## 附录 A · 关键代码位置索引

### 需求 1 · Thinker
| 位置 | 内容 |
|---|---|
| `models/minicpm_o/components/sglang_thinker.py:39` | 复用 SGLang `Qwen3ForCausalLM` |
| `models/minicpm_o/stages.py:279-339` | thinker 引擎配置（cuda graph / mixed chunk / async decode） |
| `models/minicpm_o/thinker_model_runner.py:58-75` | `capture_hidden_mode = FULL` 覆写 |
| `models/minicpm_o/thinker_model_runner.py:99-102` | per-token `detach().clone()` |
| `models/minicpm_o/thinker_model_runner.py:112-124` | `on_request_finished` 单次阻塞 `.to("cpu")` |
| `model_runner/thinker_model_runner.py:84-100` | 基类默认 `CaptureHiddenMode.NULL` + 注释 |
| `model_runner/thinker_model_runner.py:495-540` | `lookahead_eligible`：语音返回 False（:515） |
| `model_runner/thinker_model_runner.py:542-607` | `async_host_buf` / overlap launch-resolve |

### 需求 2 · Thinker→Talker
| 位置 | 内容 |
|---|---|
| `models/minicpm_o/config.py:76` | thinker `stream_to=["decode"]`（无 talker） |
| `models/minicpm_o/config.py:98-159` | talker/code2wav 独立进程放置 |
| `models/minicpm_o/routing.py:172-186` | `project_thinker_to_talker` 一次性投影 |
| `models/minicpm_o/talker_request.py:26-80` | `build_talker_request`：CPU `torch.stack`（:79） |
| `models/minicpm_o/components/sglang_talker.py:93-125` | `build_condition_embeddings`：H2D（:110-111） |
| `comm/stage_io.py:89-119` / `:122-156` | `extract_tensors`（pickle） / `extract_cuda_tensors` |
| `relay/cuda_ipc.py` | 同卡跨进程 GPU 零拷贝通道（1450 行） |
| `models/qwen3_omni/talker_scheduler.py:50-192` | 流式 handoff：partial start + 逐步门控 |
| `models/qwen3_omni/request_builders.py:860-1003` | 从 hidden/chunks 构建 talker 请求 + dual-layer |
| `models/qwen3_omni/components/talker_input.py:150-161` | `future_text_rows` FIFO 增量流 |

### 需求 3 · Flow / DiT
| 位置 | 内容 |
|---|---|
| `models/minicpm_o/components/token2wav/flow.py:46-91` | `solve_euler` 10 步 eager + per-step `cat` |
| `models/minicpm_o/components/token2wav/flow.py:153-225` | `inference`：`.tolist()` 同步 / Python 循环 pad / 逐行 scatter |
| `models/minicpm_o/components/token2wav/dit.py:69-71` | 独立 `to_q/to_k/to_v`（可 fused） |
| `models/minicpm_o/components/token2wav/dit.py:467-508` | `forward_packed`：per-step 重算不变量（:474-494） |
| `models/minicpm_o/components/code2wav.py:263-325` | `vocode`：按长度分桶 HiFT + 末尾 `.cpu()` |
| `models/auk/step_cuda_graph.py:135-152` | `AuKStepCudaGraphRunner`（Euler step 捕获模板） |
| `models/auk/fused_qk_norm_rope.py` | fused projection 参考 |
| `models/fun_cosyvoice3/config.py:19-75,162-167` | 54 个 flow CUDA graph 捕获 shape + compile/TRT 开关 |
| `models/fun_cosyvoice3/packed_dit.py:192-248` | `PackedDiT` / `RaggedRowAttention`（FA3） |
| `models/qwen3_omni/components/code2wav_cuda_graph.py:76-244` | `Code2WavCudaGraphRunner` / `GraphKey` / graph pool |

---

*文档生成于 2026-10-02，基于 sglang-omni main 分支代码级调研。行号为调研快照，实施前请复核。*
