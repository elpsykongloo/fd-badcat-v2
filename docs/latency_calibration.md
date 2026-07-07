# 现实延迟档标定 + 预注册预测表（W3 D4，裁断 A）

> 状态：**预注册定稿 2026-07-07（跑前）**。§4 预测表自此不可改（06 纪律：不符处跑前改、跑后不许动）。
> 实现：`src/latency_realistic.py`（profile 版本 `realistic-v1`）；驱动 `w2r_stream_replay.py --latency-profile realistic`
> 或零 GPU 重打分器 `scripts/w3_realistic_rescore.py`。

## 1. 为什么需要第二档（R9 → 裁断 A）

官方沙箱工具 p50 0.315s（W2 实测）没有重叠空间——机制的首响收益被基准延迟档物理压死（P3 判"判据失效"）。
裁断 A 把双档评测升格为中心主张的另一半：现实 API 延迟下，首响解耦的收益随工具时长放大。
**关键实现事实：现实档是纯记账层**——工具结果不进决策 prompt，故现实档分数是既有 trace 的确定性函数，
决策缓存 100% 复用，δ 网格 @现实档 = 零新增 GPU 决策。

## 2. 标定：κ 类 → 对数正态参数

四类（12 工具 + 3 补偿器）。参数以公开测量锚定，全部落在 06 处方带（READ 0.3–1s、写/预订 1–5s）内：

| 类 | 工具 | p50 | p95 | μ=ln(p50) | σ | 锚点 |
|---|---|---|---|---|---|---|
| `read_lookup` | get_card_benefits, get_exchange_rate, track_order, calculate_commute | 0.30s | 0.74s | ln 0.30 | 0.55 | Stripe 第三方监测 p50 120–369ms [APIContext/DORA案例]；SimpleVisa /eligibility avg 230ms/p95 310ms；arXiv:1903.07712 快簇 30–150ms |
| `read_search` | search_flights, search_apartments, search_products | 0.75s | 1.33s | ln 0.75 | 0.35 | SimpleVisa 2025-07 基准：Amadeus Flight Offers avg 680/p95 970ms、Skyscanner 740/1040、Expedia Rapid 910/1260、Booking.com 1140/1580 |
| `write_light` | update_search_filter, add_to_cart (+remove_from_cart) | 0.40s | 0.91s | ln 0.40 | 0.50 | 支付网关级写：DORA 案例端到端 p50 180/p95 420ms；SimpleVisa /application（写）avg 280/p95 360ms |
| `write_booking` | book_flight, modify_autopay, update_identity_doc (+cancel_booking, revert_autopay) | 3.00s | 5.00s | ln 3.00 | 0.31 | 旅行聚合器供方实测"200ms–3s"（OneUptime 2026-02）；GDS PNR 创建为多秒级（集成指南以秒级预算超时）；06 处方带 1–5s 的中点保守取值 |

来源 URL：
- SimpleVisa travel API benchmark (2025-07): https://simplevisa.com/travel-api-performance-benchmarks-visas-vs-flights-vs-hotels/ （供方发布，方法学公开：30 天、AWS eu-central-1、TCP→last byte）
- arXiv:1903.07712 "Benchmarking Web API Quality – Revisited"（形态学依据：重尾/双峰/分钟级离群 → 我们截断 p99.9）
- APIContext Stripe 运行情报（p50 369ms/p95 862ms）: https://apicontext.com/ai/page/api-directory-fintech-stripe.html
- DORA 合规支付基准（Stripe US p50 180/p95 420/p99 850ms）: https://dev.to/binadit/benchmarking-non-us-payment-infrastructure-a-dora-compliance-case-study-with-cloud-cost-4la8
- OneUptime 旅行聚合器监测（供方 200ms–3s）: https://oneuptime.com/blog/post/2026-02-06-monitor-travel-aggregator-api-response-opentelemetry/view

**诚实注记（R16 防御）**：预订级写调用（GDS PNR / autopay 设置）无高质量公开 p50——该类锚定于二级来源 +
06 处方带，参数在本文件预注册且官方档永远并排报告。σ=0.31 使 90% 质量落 [1.8, 5.0]s，不构造极端长尾优势。
全体类截断 p99.9（arXiv 数据显示分钟级离群，单次抽样不得支配 100 场景均值）。

## 3. 播种（确定性）

`lat = LogNormal(μ_class, σ_class)`，种子 = `sha256(example_id | fn | sorted(args) | occurrence)` 的前 8 字节
→ **per-instance RNG**：与调用顺序、线程交错、进程重启无关；两臂同 (example, fn, args) 抽同值 ⇒ 配对比较精确。
（这同时实现了 W2 备忘的 latency_injector per-instance RNG 修复——上游 PR 已取消（仓库仅本地，7/07），本地实现即终态。）
官方档播种不变（`random.seed(42)`，冻结）。

执行语义（完成锚记账）：
- **blocking = 串行**：官方 SUT 逐调用等待，`done_i = max(t_commit_i, done_{i-1}) + lat_i`；
- **TACT = DAG 并行**：`done_i = max(t_commit_i, max_parent(done)) + lat_i`（无依赖即并行；依赖边来自 `tact_dag.OpDag`）。
- 首响约定沿用 W2 冻结公式：blocking 只在结果就绪后发声（首响=完成锚）；TACT 首响 = say 锚（与工具无关）。

## 4. 预注册预测表（跑前定稿，含静默预算定律复核）

**定律复核记录**：完成锚保费 = t_commit_nominal 差 = δ 静默，与工具延迟无关 ⇒ **与档位无关** ✓ 自洽。
由此推论 (a) oracle 前沿三数（固定 123.8s / oracle 20.9s / 回收上限 83.1%，名义钟）**照搬到现实档不变**；
(b) 06 预注册版本 (iii) "链式场景 TACT ≤ blocking" 经定律复核**有条件**——并行节省必须超过窗保费——
按纪律**跑前修正**为 P-3 的条件式（这是"不符处跑前改"的实例，修正在跑前登记于此）。

| # | 预测 | 数值预注册 | 判据 |
|---|---|---|---|
| P-1 | 首响比 @现实档 | 写类场景（预订/身份/autopay 完成锚主导）blocking 首响 ≈ 1.45 + 3.0 ≈ **4.4s 级**，TACT 维持 ~1.14s ⇒ 比值 ≈ **26%**；全量 p50 比值预测 **≈45–50%** | 原 P3 判据 ≤50% @现实档达标（写类子集强达标 ≤30%） |
| P-2 | 单笔串行写完成锚保费 | TACT 完成 − blocking 完成 = **δ*（=1.5s）**，p50 全量差 ≈ 1.49s，与官方档实测一致（档位无关） | 差分 ∈ [1.3, 1.7]s 视为吻合 |
| P-3 | 链式场景完成锚 | TACT ≤ blocking **iff Σlat − max(lat) ≥ δ**。2×read_search（节省 ~0.75 < 1.5）不反超；≥2 预订写（节省 ~3.0 > 1.5）或 ≥3 调用反超。28 个链式场景中反超者为**含多写/≥3 调用的少数** | 逐场景验证条件式方向（错向 ≤2 场景） |
| P-4 | 首响-δ 解耦 @现实档 | TACT 首响在 δ 网格上平坦（ack 锚不依赖工具延迟与 δ），δ∈{0.6,1.0,1.5,2.0,2.5} 首响 p50 波动 <0.05s | 平坦性成立 |

跑后对表规则：不符处**如实分析、不改预测**；对表结果写入 W3 报告独立小节。

## 5. 复现包

`src/latency_realistic.py`（参数+播种）、本文件（标定+预注册）、`scripts/w3_realistic_rescore.py`
（对既有 result 档案的零 GPU 重打分器）、`tests/test_w3_d456.py` §A（校准回归：p50/p95 对表 ±15/20%）。
