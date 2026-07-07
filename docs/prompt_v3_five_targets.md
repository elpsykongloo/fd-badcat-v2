# Prompt v3 五靶批次（W3 D5，06 §二④定稿）

> 状态：diff 草案 2026-07-07。**尚未跑**——30 子集验证与全量复核（D6）是 GPU 项，见 `docs/w3_d456_handoff.md`。
> 实现：`src/tact_core.py` `PROMPT_V3_ADDENDUM` + `install_prompt_v3()`，由 `w2r_stream_replay.py --prompt v3`
> 或 engine_cfg `prompt:"v3"` 显式装配。默认 v2 不动（缓存键冻结）。

## 0. 纪律

- **30 调优子集**（预注册，`exp/w3/tuning30.json`）：21 个 rollback 旗标场景（机制子集，prompt 不得使其回归）
  + 9 个五靶/规范化证据场景 {travel_16, travel_21, housing_13, ecommerce_08, travel_01, travel_07, housing_06,
  finance_20, ecommerce_13}。其余 70 场景 = held-out，调优期间**不看不改**。
- v3 改变一切缓存键 ⇒ 新决策全部 GPU 生成；provider 命名 `w3p3_*`。
- 冻结后：官方档 + 现实档两条 δ 网格用终版 prompt 各重生成一次（06 §二④；--workers 12 ≈98s/全量 + judge 并发）。
- P1 判据从此差分制（≥−1pt vs blocking 0.570），绝对值不设防。
- 诚实注记（原样进报告）：靶① 修复会抬高低 δ 曲线（漏调类可救），但官方轨 wrong-arg 已提交即死不受影响——窗必要性叙事无恙。

## 1. 五靶与 diff

规则 10–14 追加于 PROMPT_V2 之后（文本见 `src/tact_core.py`，此处给依据与预期翻转）：

| # | 规则 | 证据（w3_ledger） | 预期可救 | 风险 |
|---|---|---|---|---|
| 10 后提交瘫痪 | 快照示已执行错参时：不许只嘴上更新——re-launch 修正调用（READ 类），且其余请求照常发 op | §4B：3/3（eco19/hou25 @δ≤0.6 漏调、fin12b @δ≤1.0 不重发）；分类台账 missing_call TACT 7 例 | 低 δ 曲线抬升；missing_call 类部分回收 | 过度 re-launch → dedup 幂等已兜底 |
| 11 宣告即撤销 | 修订已宣告、新值未到 ⇒ 立即 cancel pending op，零成本；新值到再 launch | §4C：travel_10 EoU1 错失 cancel（静默预算 3.91 > 全网格，唯一零保费救援路径）；oracle 前沿 travel_10 救援/撤销成本比 2.61× | travel_10 全网格翻转 | 误伤犹豫语气 → 仅在"明确宣告变更"触发，few-shot 界定 |
| 12 patch 打对字段 | patch 只改用户更正的字段；对照每个 arg 现值判断被否定者（origin/destination 是常见混淆）；diff 不复述未变字段 | housing_17b：EoU2 patch origin='here'（应 'my house'），后续只 patch mode 永不改回 | hou17b（当前唯一 TACT 输夹） | 无：约束性规则 |
| 13 自我打断替换 | "check X— actually never mind, do Y"：X 弃、Y 是真请求必须 launch | §4G：finance_23 双臂从不调用 modify_autopay（句式系统性回避）；W1 legacy 曾判过 | fin23（共通失分，两臂同救） | 无副作用：Y 本来就该发 |
| 14 实体规范形态 | 全称地名（Las Vegas）、基数日期（June 3）、紧凑 ID（DL555）、无冠词、无所有格、名词不加形容词 | travel_19（3rd/3）、travel_16/21（Vegas）、裁断 C 四例；分类台账 canonicalization 6 例 + format 24 例（后者规范化器已救，prompt 双保险） | travel_16 全量 +1；exact 轨若干；与 norm-v1 双保险 | 与用户逐字复述冲突场景 → 官方 gold 本身是规范形态，方向一致 |

## 2. 验证协议（GPU，交接）

1. 30 子集 @官方档 δ*=1.5 + blocking 对照，`--prompt v3 --ids-file exp/w3/tuning30.json`；
   门：rollback 12/17 不降、五靶证据夹按上表方向翻转、无新增回归 >1 夹。
2. 迭代若需（最多 2 轮，只看 30 子集）。
3. 冻结 v3 → 全量 100 双臂 A 档一次（P1 差分制对账）→ 双档 δ 网格重生成 → HumDial 门（Δ≥−1，同日对称判）。
4. 蓝图#8 试点素材：分类台账 asr_mishear 已 TRIGGERED（TACT 4 例/blocking 6 例）——是否立项 W3 最小试点
   （Phase-B 标记不确定槽 → 引擎带焦点提示重听）待用户/神谕裁断；prompt v3 不含此项。
