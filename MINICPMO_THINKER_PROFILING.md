# MiniCPM-o 4.5 Thinker 阶段性能剖析记录（对应 Issue #2284 需求一）

本文记录一次针对 MiniCPM-o 4.5「Thinker 阶段」的性能剖析（profiling）：做了什么测量、用的是什么方法、
测到的数字、以及由此得出的结论与建议。对应 Issue #2284 中这一条需求：

> Thinker: profile prefill/decode scheduling, hidden-state capture and per-token copies;
> tune existing SGLang batching/graph/async-decode capabilities where measured.

本次只做了「非侵入式」的测量：没有重启服务、没有修改任何默认配置、没有提交任何代码。
测量进行到「发现阶段」结束就停下，尚未进行需要重启服务的对照实验（原因见第 9 节）。

---

## 1. 结论摘要（先给要点）

1. **Thinker 的 prefill（处理输入 prompt）很快**，大约 12 毫秒，不是瓶颈；耗时集中在 decode（逐个生成 token）阶段。
2. **decode 阶段确实一直在用 CUDA graph**（把一串 GPU 操作录制后重复回放的加速机制）。
   服务日志里 526 条 decode 记录全部是 `cuda graph: True`，没有一条回退到不使用 graph 的 eager 模式。
   所以「捕获 hidden states 会导致 graph 失效」这个担心，在本部署里不成立。
3. **原以为的 Thinker 内部开销其实很小**：为语音生成而捕获 hidden states（thinker 每一步产生、用来给
   talker 做语音条件的中间向量）、每生成一个 token 复制一次、以及关闭 overlap（异步重叠 decode）——
   这三项加起来，在单请求、无干扰的情况下只让每个 token 的 decode 时间增加约 **2.7%**。
4. **语音请求真正的两个瓶颈是**：
   - **显存耗尽（out of memory，OOM）**：在 32GB 单卡上，语音请求在并发数达到 2 时就开始报显存不足，
     稳定可用的并发数只有 1 到 2。
   - **各阶段串行、且共用一张卡互相争抢**：一次语音请求里，talker（生成语音 codec token 的阶段）
     占了端到端时间的约 63%，而且它必须等 thinker 完全结束才开始；并发升高时 thinker、talker、
     code2wav（把 codec token 转成波形的阶段）同时抢 GPU，使 thinker 的每 token decode 时间上升约 39%。
5. **因此建议调整优化优先级**：Thinker 内部那点隐藏态捕获的微开销（约 2.7%）不值得优先优化；
   真正该先动的是「显存预算」和「让 thinker 与 talker 重叠起来（即 Issue #2284 需求二的流式交接）」。

---

## 2. 环境与版本记录

| 项目 | 值 |
|---|---|
| 代码仓库 commit | `ca99cb975e6da2ef8f4ee2a5064894d51579b1af`，工作区干净（无未提交改动） |
| sglang 版本 | 0.5.20 |
| torch 版本 | 2.11.0+cu130 |
| Python | 3.12（虚拟环境在 `/venv/main`） |
| GPU | 单张 NVIDIA RTX 5090，32GB（实际可用 31.36 GiB） |
| 模型权重 | `/cpm-workspace/models/MiniCPM-o-4_5`，离线加载（环境变量 `HF_HUB_OFFLINE=1`） |
| 服务进程 | 已在运行（进程号 40449），启动命令见 `DEPLOYMENT_MINICPMO_LOCAL.md` 第 4 节，含 `--thinker.engine.mem_fraction_static 0.80 --talker.engine.mem_fraction_static 0.10` |

各阶段所在的进程（都在 GPU 0 上）：

- 进程 40497：preprocessing、**thinker**、image_encoder、audio_encoder、decode（这几个阶段共用一个进程）
- 进程 40498：**talker**
- 进程 40499：**code2wav**

剖析工具的可用性：

- `py-spy`（一个通过读取进程内存来采样 Python 调用栈的工具）已安装 0.4.2 版，但**无法使用**：
  容器缺少 `CAP_SYS_PTRACE` 这项能力（读取其他进程内存所需的系统权限）。即使以 root（超级用户）身份运行，
  读取内存的系统调用仍然返回「权限被拒绝」（permission denied）错误。
- `nsys`（NVIDIA 的性能剖析工具）已安装。
- `dcgmi`（NVIDIA DCGM 的命令行工具，可采样 GPU 的 SM 活跃度等硬件指标）**未安装**。

---

## 3. 测量方法

### 3.1 两组对照负载

对同一个正在运行的服务，发两种请求做对照（都设 `max_tokens=128`）：

- **仅文本输出请求**：`modalities=["text"]`。这条路径上，thinker 的 overlap（异步重叠 decode）是开启的，
  也不捕获 hidden states。作为「没有语音开销时」的基准。
- **文本加语音输出请求**：`modalities=["text","audio"]`。这条路径上，thinker 会以 FULL 模式捕获 hidden states
  （用来给 talker 做语音条件），并且对这类请求 overlap 是关闭的。

每个请求的输入都带一个随机唯一前缀，目的是绕过前缀缓存（radix cache，对相同输入前缀复用已算结果的机制），
否则重复请求会命中缓存、跳过 prefill，测出来的就不是真实的 prefill/decode 开销。

负载由一个临时的最小驱动脚本发起（直接调用 `/v1/chat/completions` 接口）。仓库自带的
`benchmarks/eval/benchmark_omni_seedtts.py` 没有使用，因为它依赖未安装的 `jiwer` 库，并且需要下载 seed-tts 数据集。

### 3.2 每请求分阶段计时（用内置的事件记录器）

服务内置了一个「每请求事件记录器」：各阶段进程会把带时间戳的事件写进 JSONL 文件，
事后按 request_id 合并、渲染成报表。开启方式是调用 HTTP 接口 `POST /start_request_profile`
（指定一个输出目录），结束后调用 `POST /stop_profile`。渲染用 `python -m sglang_omni.profiler <目录> --format table`。

用到的关键事件名与由此算出的区间：

- prefill 耗时 = `scheduler_prefill_start` 到 `scheduler_prefill_end`
- decode 耗时 = `scheduler_prefill_end` 到 `stage_complete`
- 每个 token 的 decode 时间 = decode 耗时 ÷ 该请求生成的 token 数（token 数取自响应的 `usage.completion_tokens`）
- 阶段之间的交接延迟 = `stage_hop_sent` 到下一阶段的 `stage_input_received`

### 3.3 并发数对比测试

分别在并发数 1、2、4 下重复上面的负载（并发数 = 同时在处理的请求数），记录吞吐、延迟、错误数和 GPU 利用率。
语音请求没有测到更高并发，因为并发 2 起就开始报显存不足（见第 5.5 节）。

### 3.4 GPU 利用率采样

负载运行期间，用 `nvidia-smi` 每隔约 0.2 秒读一次 `utilization.gpu`，取平均值。
需要注意：`utilization.gpu` 只表示「采样这一刻是否有 GPU kernel 在运行」，并不代表 GPU 被用满的程度；
而且语音请求期间 thinker、talker、code2wav 共用一张卡，这个数字是三者合在一起的，无法单独归给 thinker。

### 3.5 未能进行或主动跳过的测量

- **基于 py-spy 的调用栈采样**：因容器缺少 ptrace 权限而无法进行（见第 2 节）。改用「每请求事件记录器 +
  服务日志里的调度指标 + 并发数 1 与并发数 4 的隔离对比」来达到同样的归因目的。
- **torch trace / GPU kernel 级归因**：主动跳过。因为并发数 1 的隔离对比已经把隐藏态捕获相关开销
  界定在约 2.7%（很小），而服务日志已经能确定 CUDA graph 是否生效；在这种前提下再做 kernel 级的细粒度归因
  收益很低，还会产生较大的 trace 文件（本机磁盘只剩 8GB）。

---

## 4. 隔离对比的思路（本次最关键的一个设计）

在并发数 1 下，一条语音请求的处理是严格串行的：thinker 先 decode 完，talker 才开始，然后是 code2wav。
也就是说，**thinker 在 decode 的时候，talker 和 code2wav 是空闲的**，不存在抢 GPU 的问题。

因此：

- **并发数 1** 下测到的「语音 vs 仅文本」的每 token 差异，反映的是**隐藏态捕获本身的开销**
  （捕获 + 每 token 复制 + 关闭 overlap + 请求结束时一次性拷回 CPU）。
- **并发数 4** 下多出来的差异，反映的是**多个请求同时在飞时，talker / code2wav 与 thinker 争抢同一张 GPU** 的影响。

这样就能把「Thinker 内部开销」和「跨阶段抢卡」这两件事分开，避免把它们混为一谈。

---

## 5. 测量结果

### 5.1 Thinker 各阶段耗时（并发数 4，各 24 个请求）

| 请求类型 | prefill 平均 | decode 平均 | 平均生成 token 数 | 每 token decode 时间 |
|---|---|---|---|---|
| 仅文本输出 | 12.3 毫秒 | 1136 毫秒 | 112.46 | 10.10 毫秒 |
| 文本加语音输出 | 17.1 毫秒 | 1453 毫秒 | 103.42 | 14.05 毫秒 |

排队等待、请求构建等其它环节都在 1 毫秒以下，可忽略。

### 5.2 并发数 1 的隔离对比（各 10 个请求）—— 关键结果

| 请求类型 | prefill 平均 | decode 平均 | 平均生成 token 数 | 每 token decode 时间 |
|---|---|---|---|---|
| 仅文本输出 | 11.9 毫秒 | 1092 毫秒 | 109.8 | 9.95 毫秒 |
| 文本加语音输出 | 12.8 毫秒 | 1043 毫秒 | 102.11 | 10.22 毫秒 |

- 并发数 1 时，语音请求的每 token decode（10.22 毫秒）与仅文本（9.95 毫秒）几乎一样，**只高约 2.7%**。
- 并发数 4 时，语音（14.05 毫秒）比仅文本（10.10 毫秒）**高约 39%**。
- 对照来看，仅文本请求从并发数 1 到并发数 4，每 token 时间几乎不变（9.95 → 10.10 毫秒），
  因为它没有 talker / code2wav 在抢卡。

**结论**：语音请求在并发时每 token 多出来的那约 39%，主要来自与 talker / code2wav 争抢 GPU，
而不是隐藏态捕获本身的开销（后者在并发数 1 下只有约 2.7%）。

### 5.3 CUDA graph 是否生效

统计服务日志里的 decode 批次记录：`cuda graph: True` 出现 **526 次**，`cuda graph: False` 出现 **0 次**。
说明 decode 一直在用 CUDA graph，语音路径也不例外；以 FULL 模式捕获 hidden states 并没有让它退回到 eager 执行。

### 5.4 并发数对比测试（每个并发点 12 个请求）

仅文本输出请求：

| 并发数 | thinker 吞吐（token/秒） | 延迟 p50 | 错误数 | GPU 利用率均值 |
|---|---|---|---|---|
| 1 | 99.2 | 0.99 秒 | 0 | 95.6% |
| 2 | 188.0 | 1.10 秒 | 0 | 99.2% |
| 4 | 350.4 | 0.99 秒 | 0 | 98.7% |

文本加语音输出请求：

| 并发数 | thinker 吞吐（token/秒） | 延迟 p50 | 错误数 | GPU 利用率均值 |
|---|---|---|---|---|
| 1 | 34 到 39 | 2.12 秒 | 0 | 70.4% |
| 2 | 41.7 | 4.74 秒 | 2（共 12 个） | 83.7% |
| 4 | 56.6 | 5.39 秒 | 3（共 12 个） | 87.8% |

说明：

- 仅文本请求随并发数近似线性扩展（吞吐 99 → 188 → 350），延迟基本不变，说明批处理（batching）与 overlap 工作良好。
- 语音请求几乎不扩展（吞吐 34 → 42 → 57），延迟大幅上升（2.1 → 4.7 → 5.4 秒），并且从并发数 2 起出现显存不足错误。
- 语音请求这一列的「thinker 吞吐」是按「thinker 生成的文本 token 数 ÷ 整个请求的墙钟时间」算的，
  而整个请求时间还包含了后面 talker 和 code2wav 的时间，所以这个数字被稀释了、偏低；
  thinker 自身真实的 decode 速度应看第 5.2 节的分阶段数据。

### 5.5 显存耗尽（out of memory）

语音请求在并发数 2 及以上会触发 CUDA 显存不足错误（并发数 1 偶尔也会）。错误信息节选：

```
CUDA out of memory. Tried to allocate 262.00 MiB. GPU 0 has a total capacity of 31.36 GiB
of which 58.94 MiB is free. Process 40498 has 3.96 GiB memory in use.
Process 40497 has 23.52 GiB memory in use. Including non-PyTorch memory,
this process has 3.80 GiB memory in use.
```

即：thinker 所在进程（40497）占 23.52 GiB，talker 进程（40498）占 3.96 GiB，code2wav 另占约 3.8 GiB，
合计约 31.3 GiB，逼近 31.36 GiB 的上限，只剩几十 MiB 空闲，于是稍大一点的分配就失败。
当前 0.80 / 0.10 的显存分配比例，加上「每个在飞请求的 hidden states 会一直留在 GPU 上、直到该请求结束才拷回 CPU」，
共同导致没有余量。错误信息本身也建议尝试环境变量 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 来减少显存碎片。

### 5.6 端到端语音请求的各阶段占比（并发数 4）

一条语音请求约 5.5 秒，各阶段依次为：

- thinker decode：约 1453 毫秒（约占 26%）
- thinker 到 talker 的交接：约 1.6 毫秒
- talker decode：约 3483 毫秒（约占 63%）
- code2wav：约 521 毫秒（约占 9%）

交接本身的数据传输很便宜（1.6 毫秒），真正的问题是 talker 必须等 thinker 完全结束才开始，
两者没有重叠，于是 talker 的 3.5 秒完全串在 thinker 的 1.5 秒之后。

### 5.7 GPU 利用率

仅文本请求 96% 到 99%，语音请求 70% 到 88%（随并发上升）。
但如第 3.4 节所述，这个数字对语音请求来说是 thinker、talker、code2wav 合在一起的，
不能单独说明 thinker 是否把 GPU 用满，因此不作为 thinker 层面的结论依据。

---

## 6. 对四个待验证假设的结论

剖析前，从代码里提出了四个假设，逐一给结论：

- **假设一：语音请求关闭了 overlap（异步重叠 decode）。**
  代码属实——共享基类的 `lookahead_eligible` 函数在请求要求输出音频时返回 False。
  但实测每 token 影响很小（并发数 1 下约 2.7% 里的一部分）。它更可能影响的是「批处理能否随并发扩展」，
  而这一点又被显存上限和抢卡问题掩盖了。
- **假设二：每生成一个 token 就在 GPU 上复制一次 hidden states。**
  代码属实（`hidden.detach().clone()`）。并发数 1 下开销可忽略（包含在 2.7% 内）。
- **假设三：请求结束时把整段 hidden states 一次性、阻塞式地拷回 CPU。**
  代码属实（`torch.stack(seq).to("cpu")`）。每个请求只发生一次，数据量约 700 KB，耗时在亚毫秒级，可忽略。
- **假设四：以 FULL 模式捕获 hidden states 会让 CUDA graph 失效、退回 eager。**
  **证伪**——日志显示 decode 全程使用 CUDA graph（526 次 True，0 次 False）。

---

## 7. 结论：真正的瓶颈在哪里

- Thinker 内部与隐藏态捕获相关的开销（假设一、二、三合计）在单请求下只有约 2.7%，**不是主要矛盾**。
- CUDA graph 已经在 decode 上生效（假设四证伪），**无需处理**。
- 语音路径真正受限的两点是：
  1. **显存不足**：把稳定可用的语音并发数压在 1 到 2，是当前最硬的天花板。
  2. **各阶段串行 + 同卡争抢**：talker 占端到端约 63% 且串在 thinker 之后；并发升高时三个阶段抢一张卡，
     使 thinker 每 token decode 时间上升约 39%。

---

## 8. 建议（按优先级）

1. **优先解决显存问题**（收益最直接）。可尝试：设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`；
   把 thinker 的 `mem_fraction_static` 从 0.80 降到 0.75 左右，给 talker / code2wav 留余量；
   以及考虑把 hidden states 增量地拷回 CPU，而不是在 GPU 上攒到请求结束。目标是把稳定语音并发数从 1 到 2 抬上去。
2. **让 thinker 与 talker 重叠起来**（端到端收益最大）。这对应 Issue #2284 需求二的「流式交接」：
   让 talker 在 thinker 还没结束时就逐步开始，从而把当前串行的 1.5 秒 + 3.5 秒重叠掉。
3. **降低 Thinker 内部微优化的优先级**。把每 token 的复制改写成 pinned ring buffer、或给语音请求强行打开 overlap，
   实测每步收益只有约 2.7%，投入产出比低。
4. **CUDA graph 保持现状**，decode 已在用，无需改动。

---

## 9. 可选的后续验证实验（需要重启服务，等待确认）

下面的每个实验都要重启服务（每次约 5 到 8 分钟）并改启动配置，属于对照实验（A/B：只改一个变量、其余不变；
必要时先做 A/A，即两边都用当前配置，用来测量噪声波动范围，再判断改动效果是否显著）。因为涉及重启，暂停在此等待确认：

1. 重启时加环境变量 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，测量显存不足错误率与最大稳定语音并发数。
2. 重启时把 thinker 的 `mem_fraction_static` 从 0.80 改为 0.75（可选把 talker 从 0.10 改为 0.12），
   测量「KV 缓存变小」与「显存不足缓解」之间的权衡点。
3.（优先级低）重启时在语音路径上关闭 thinker 的 CUDA graph，用来反向确认 graph 的价值（预期会变慢）。

补充：给语音请求打开 overlap 属于**改代码**（改 `lookahead_eligible` 的判断），不是命令行开关能做的对照实验；
而且实测每步收益低，建议放到后续优化阶段，不在本次对照实验范围内。

---

## 10. 运行状态与清理

- 全程没有重启服务、没有改任何默认配置、没有提交代码。
- 每次测量后都调用了 `POST /stop_profile` 关闭事件记录（返回 `{"run_id": null}` 表示已无进行中的记录），
  所有测量结束后服务健康检查正常。
- 测量产生的中间数据（事件 JSONL、负载日志）都很小（合计约 600 KB），没有生成体积较大的 torch trace 或 nsys 文件。
- 基于 py-spy 的调用栈采样在本容器无法进行（缺少 ptrace 权限）；若日后需要对「抢卡造成的空隙」做更细的归因，
  可在一个允许 ptrace 的主机上补做。

---

## 11. 术语对照表

| 术语 | 含义 |
|---|---|
| thinker / talker / code2wav | MiniCPM-o 的三个推理阶段：thinker 是语言主干，talker 生成语音 codec token，code2wav 把 codec token 转成音频波形 |
| prefill | 处理输入 prompt、填充 KV 缓存的阶段 |
| decode | 逐个生成输出 token 的阶段 |
| hidden states | thinker 每一步产生、用来给 talker 做语音条件（conditioning）的中间向量 |
| CUDA graph | 把一串 GPU 操作录制下来、之后重复回放的加速机制，可减少 CPU 逐个启动 kernel 的开销 |
| eager（模式） | 不使用 CUDA graph、逐个操作即时执行的普通模式 |
| overlap / async decode | SGLang 把「调度下一步」与「执行当前步」重叠起来的异步 decode 机制 |
| concurrency（并发数） | 同一时刻正在处理的请求数量 |
| batching（批处理） | 把多个请求的 decode 步骤合并到一次 GPU 计算里 |
| out of memory（OOM，显存耗尽） | GPU 显存不足导致分配失败的错误 |
| utilization.gpu | `nvidia-smi` 报告的「采样时刻是否有 kernel 在运行」的粗粒度利用率，不代表 GPU 被用满的程度 |
| prefix cache / radix cache | 对相同输入前缀复用已计算结果的缓存 |
| KV 缓存 | 注意力机制里缓存的 key/value，占用显存，容量影响可支持的并发与上下文长度 |
| ptrace / CAP_SYS_PTRACE | 读取或调试其它进程内存所需的系统能力；本容器缺少它，导致 py-spy 无法附着到进程 |
| A/B 对照实验 | 只改一个变量、其余保持不变的对比实验；A/A 指两边都用基线配置，用来测量噪声波动范围 |
| p50 / p95 | 延迟的中位数 / 95 分位数 |
| 交接（handoff） | 一个阶段把结果传给下一个阶段的过程 |

---

*本文基于 2026-10-02 在单张 RTX 5090、sglang 0.5.20、torch 2.11.0+cu130、代码 commit `ca99cb97` 上实测。*
*文中行号与日志数字为本次测量快照，后续如改动代码或环境需重新核对。*
