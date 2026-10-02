# 需求一（Thinker）Profiling 执行计划

## Summary
需求一 = "profile prefill/decode 调度、hidden-state 捕获与 per-token 拷贝；在实测基础上调优 SGLang 现有 batching/graph/async-decode"。本计划把它落到仓库既有工具上：`METHODOLOGY.md` 的 5 层方法 + `sglang_omni/profiler/` 的 per-request 事件记录器 + `omni-gpu-deep-dive` skill，驱动脚本用 `benchmarks/eval/benchmark_omni_seedtts.py`。分两个阶段：**发现型测量（非侵入）** → 确认门 → **Layer 4 A/B 调优 + 回归**。全程只测量、只用 CLI override，不改代码默认值、不提交。

## 待验证假设（均来自代码，profiling 要给出行号级证据）
- H1 语音 decode 同步、无 overlap：`model_runner/thinker_model_runner.py:515` `lookahead_eligible` 在 `should_generate_audio_output` 时 `return False`。
- H2 per-token GPU 拷贝开销：`models/minicpm_o/thinker_model_runner.py:99-102` 每步 `hidden.detach().clone()`。
- H3 末尾阻塞式 D2H：`thinker_model_runner.py:112-124` `torch.stack(seq).to("cpu")`（非 pinned）。
- H4 FULL hidden 捕获下 CUDA graph 是否真 replay 还是退化 eager：`thinker_model_runner.py:58-75` 覆写为 `CaptureHiddenMode.FULL` vs 基类默认 `NULL`（`model_runner/thinker_model_runner.py:84-100` 注释）。

## 阶段 0 · 环境与预检（METHODOLOGY §1）
- 记录基线指纹：`git rev-parse HEAD`、`git status --short`、`pip show sglang torch`（应为 sglang 0.5.20 / torch 2.11.0+cu130）、GPU=RTX 5090、checkpoint=`/cpm-workspace/models/MiniCPM-o-4_5`（本地目录 + `HF_HUB_OFFLINE=1`，无 refs/main 漂移风险）。
- GPU 干净检查：`nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv`（util 与 memory 都看）。
- 起服务（沿用 `DEPLOYMENT_MINICPMO_LOCAL.md` 第 4 节命令，thinker 0.80 / talker 0.10），额外导出 `SGLANG_TORCH_PROFILER_DIR=<run 目录>`；mapping 侧再单独起一份带 `SGLANG_TORCH_PROFILER_WITH_STACK=1`。
- 确认驱动脚本能力：`python benchmarks/eval/benchmark_omni_seedtts.py --help`，核对是否有 `--use-existing-server`/`--profile-events`/`--profile-event-dir`/`--sample-util`/`--warmup`/并发项（§4 规则：先看 --help，避免脚本自己再起一个服务撞端口）。缺 `--profile-events` 时改用 profiler 控制面（`/start_profile` 带 run_id）开启事件记录器。
- 发 1 条真实语音请求（`modalities:["text","audio"]`）确认端到端可用后再压测（§1 末条）。

## 阶段 1 · 发现型测量（非侵入，Layer 1→2→3 + 事件记录器 + deep-dive）
关键设计：**语音路径 vs 纯文本对照**双臂。纯文本臂 overlap 开、无 hidden 捕获；语音臂 overlap 关、FULL 捕获。两臂 per-token 差值 ≈ H1+H2 的"语音税"。

1. Per-request 阶段分解（最低开销、最直接，先做）
   - 开启事件记录器，跑语音负载；用 `python -m sglang_omni.profiler` / `sglang_omni/profiler/views.py` 渲染。
   - 取区间（`views.py:164-172`）：prefill=`scheduler_prefill_start→scheduler_prefill_end`；decode≈`scheduler_prefill_end→stage_complete` 与 `scheduler_first_emit→stage_complete`；thinker TTFT=`scheduler_prefill_start→stage_first_stream_chunk_sent`。
   - 由 decode 区间 / 输出 token 数得**每 token decode 墙钟**，语音臂 vs 文本臂对比 → 量化 H1+H2。hop（`stage_hop_sent→stage_input_received`）顺带为需求二打底。
2. Layer 1 GPU busy ratio（判 GPU-bound 还是 CPU/编排-bound）
   - Kernel-timing：`/start_profile`(enable_torch) → 压测 → `/stop_profile`，导出 chrome trace，对 `kernel+GRAPH_TRACE+MEMCPY+MEMSET` 取**区间并集**/墙钟。
   - 本环境（torch 2.11+cu13）图 replay 落在 `cat=="cuda_runtime"` 的 `cudaGraphLaunch`，**不是** `GRAPH_TRACE`（§Layer1 已在本仓库验证）→ 用 `cudaGraphLaunch` 是否存在判定 H4（语音臂 graph 是否真 replay）。
   - 有 `dcgmi` 则并行 `dcgmi dmon -e 1002,1004 -d 100 -i 0`，紧贴压测窗口交叉验证。
3. Layer 2 py-spy（若 Layer 1 判为 CPU/编排-bound）
   - 定位 thinker 引擎/scheduler 子进程（非 HTTP 顶层），`py-spy dump --pid` 认线程；`py-spy record --format raw --idle --subprocesses --pid <thinker> --duration 25 --rate 20`（先 rate 5 试lag）。
   - 按 file:line 归组、聚焦 scheduler 线程（`_run_scheduler`）；找 `post_process_outputs`/`clone`/`torch.stack`/`.to(` 叶子帧（H2/H3）与同步等待帧（H1）。≥3 次重复看是否收敛。
4. omni-gpu-deep-dive（把 GPU 时间归因到 python 行，验 H2/H3 kernel 级）
   - 对 thinker stage 抓 mapping/formal trace 对（mapping：graph off + with_stack；formal：真实服务配置 graph on）。
   - 读 kernel 表 "Python location" 是否把小 copy kernel / D2H MEMCPY 归到 `thinker_model_runner.py:102`/`:123`；overlap 表 gaps 印证同步 decode 空档。thinker 为 Qwen3 dense，用真实权重。
5. Layer 3 并发扫描
   - 并发 {8,16,32,64}（thinker `max_running_requests=64`，`stages.py:299`），语音/文本各扫，2–3 重复 + 1 warmup，采 util + 吞吐/延迟。看语音臂 util 是否随并发平台化/下降（H1 预期：每步同步 + per-token clone 是固定税）。

## 阶段 2 · 确认门（discovery → 选 A/B 变量）
汇总阶段 1：给出 (a) 语音路径 GPU-bound 还是 CPU/编排-bound；(b) 每 token 语音税量级；(c) graph 是否真 replay（H4）；(d) 主导叶子帧/kernel。据此**人工确认**下一步 A/B 单变量，未确认不进 Layer 4（对齐 model-profiling skill 的两次 pause 设计）。

## 阶段 3 · Layer 4 A/B 调优 + Layer 5 回归（"tune where measured"）
- 先 A/A 标定噪声地板（共享主机用交错配对），再单变量 A/B（仅 CLI override，双臂 warmup/GPU 环境一致）：
  - `--thinker.engine.disable_cuda_graph true/false`（语音臂）→ 直接量化 H4：FULL 捕获下 graph 到底值不值。
  - `--thinker.engine.chunked_prefill_size`（默认 8192，`stages.py:303`）、`enable_mixed_chunk`（:302）→ prefill 侧。
  - `--thinker.engine.async_decode_min_batch_size`（默认 2）→ 主要影响文本臂（语音臂被 H1 强制同步，改它无效，此点本身即结论）。
- 说明：让**语音路径吃上 overlap** 需改代码（pinned ring buffer 解耦捕获/读取竞态），属优化阶段，不在 profiling 内改；profiling 只负责用数据证明其收益上限。
- Layer 5：任一改动跑功能回归，至少含一条长输出 + 一条结构不同样本（多轮/带图像+音频输入，走不同 chunked-prefill 路径），确认文本与音频质量无回退。

## 阶段 4 · 产出与追踪
- 本地：`.profiling-runs/minicpm_o/profile.md`（gitignored）+ 原始件（trace、py-spy raw、events_*.jsonl、benchmark 日志）。按层写 方法/发现/证据强度（§3 item8 rubric），含 findings-evidence-recommendation 表；每个数字可回溯到命令 + 原始件路径。
- 追踪：结论回灌 issue #2284 的 Thinker 条目；按 model-profiling skill 规范在 tracking issue #1798 下开 sub-issue（附小体积原始件）。**不自动提交**；唯一可提交物是 `METHODOLOGY.md` 的通用教训补充，交用户 review。

## 工具/命令索引（本计划用到的仓库既有资产）
- 方法论：`.claude/skills/model-profiling/METHODOLOGY.md`（§1 预检、§2 Layer1-5、§4 工具）。
- GPU 行级归因：`.claude/skills/omni-gpu-deep-dive/SKILL.md` + `scripts/omni_trace_pair.py`。
- 事件记录/视图：`sglang_omni/profiler/event_recorder.py`、`views.py`、`__main__.py`、`torch_profiler.py`、`profiler_control.py`。
- 负载驱动：`benchmarks/eval/benchmark_omni_seedtts.py`（长输出/流式可加 `benchmark_omni_streaming_ttft.py`）。
- CPU 隔离（共享主机）：`.github/scripts/pin_to_ci_cpuset.sh`、`examples/mps_dp/{launch,autodp}.sh`。

## 验收标准（"measured" 判定）
profiling 视为完成当且仅当对 H1–H4 各给出带证据强度分级的结论（含负结论如"graph 已生效/该 knob 无需改"），量化语音路径每 token decode 税，并明确：是否存在单变量、无回归、超过噪声地板的调优收益；据此决定是否立项"hidden 捕获重写（pinned 增量拷贝）+ 语音 overlap"。

## 假设与边界
- 假设可在本机重启服务并设 profiler 环境变量；单卡 RTX 5090，thinker/talker/code2wav 同卡，压测时注意三者争用。
- 事件记录器 marker 名以 `views.py:164-172` 为准；benchmark 具体 flag 以 `--help` 为准（阶段 0 核对）。
- 行号为调研快照，执行前复核。