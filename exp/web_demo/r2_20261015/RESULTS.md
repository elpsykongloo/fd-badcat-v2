# r2：路由 prompt 修“等一下”自答（2026-10-15）

## 结论

采用 wr3：只在 `input_route` 的 `stop_only` 定义后追加一句“用户说明自己要暂时离开、先去做别的事……也是stop_only：用户回来后会再开口。”（含“稍等，我去开个门”“等一下，我接个电话”“Hold on, someone's at the door”三个例句）。`configs/demo_chat.yaml` 已改为 wr3，与实验 prompt 逐字一致。

- 问题：用户说“等一下，我接个电话”后被判 `yield_wait`，2 s 等待释放后助手自答。根因是路由标签，不是释放时长。
- HumDial 测试集（4999 条，只做最终验收）：总体 76.9→78.6（+1.7，130 条变好／45 条变坏）；中文 wait 71.7→85.0（自答 83→43），英文 wait 63→75，中文 pause 49→55.3。
- 代价：中文 `others_talk_to_user_before` 28→24、英文 16→13.3；其余各类 ±1 pp；首响中位不变。

## 候选与开发集

train/dev 用于选 prompt；wr0 是无关措辞扰动，作噪声基准。

| 版本 | 改法 | train 中文停后静默 | dev 中／英 | 主要副作用 | 判定 |
|---|---|---|---|---|---|
| base | 原 prompt | 0.025 | 0.46／0.41 | — | 基线 |
| wr1／wr2 | 改 `yield_wait` 定义 | 0.91／0.895 | — | 重复请求中文 −17／−10.5 pp | 否决 |
| wr3 | `stop_only` 后加暂离句 | 0.65 | 0.78／0.73 | train 重复中文 −5.5、dev 重复英文 −6、dev `tp_before` −8；pause +4 | 采用 |
| wr4 | 只加例句 | 0.04 | — | 无效 | 否决 |
| wr5 | 改写为“好去做别的事” | 0.255 | — | 重复中文 −15 | 否决 |
| wr6 | wr3＋“等等后接提问不算” | 0.625 | 0.76／0.72 | 重复中文 −9～−17 | 否决 |
| wr0 | 无关扰动 | 0.025 | — | 各类 ±1.5 pp（英文 pause +6） | 噪声基准 |

副作用机制：prompt 在 `stop_only` 附近提到“等等”，会让“等等，刚才没听清楚。”这类半句被提前判 `yield_ready`。wr3 剩下的失败几乎都是“等一下，我先记个笔记”。

## 真人数据

- 整场回放（10-09～10-12 的 11 场，各跑 2 遍）：`speech_start` 132／132、`wait_released` 11／11、`keep` 37／38、`yield_wait` 64／71，逐场差异与重复跑的随机翻动同量级，无系统差异。
- 逐次路由 A/B（30 场真人会话的 744 次 `input_route` 采样，只换系统 prompt，音频与上下文不变）：

| 对比 | 翻动 | `yield_ready→yield_wait` | `yield_wait→yield_ready` | `yield_ready→keep` |
|---|---|---|---|---|
| base 对 base 重跑（噪声） | 36 | 5 | 12 | 12 |
| base 对 wr3 | 58 | 29 | 10 | 11 |

  超出噪声的只有 `yield_ready→yield_wait`（+24），逐条看几乎都是话没说完的截断句（如“……我会觉得说这个”“……然后怎么”），改判等待是对的；其余方向与噪声持平。

## 上线检查

- `pytest`：1065 passed（wr3 配置）。
- `scripts/check_guarded_turns.py`：rc=0。
- 首响基准（`lat_bench/report.md`，40 ms 单程注入，与 c3_combo 同参数）停嘴→开播 P50：short 1091 对 1099、long 1273 对 1253、real 1260 对 1266、real_first 1146 对 1122 ms，持平。cont 的 R P90（569 对 223）与 history（n=4，1418 对 1116）变慢，这段合成用例跑在 08:56～09:06 UTC，正好有两场线上会话同时占用 Omni，按并发干扰处理。第一次跑因上线脚本重启 Omni 被打断，作废（`r2_wr3.run2.log`）。

## 文件

- `make_arms*.py`、`route_wr*.txt`：候选 prompt；`hd_eval/arms_wr*.json`：评测臂。
- `r2a.sh`～`r2h.sh`：各轮评测链；`hd_eval/{train,dev}/*_summary.*`：开发集汇总；`hd_test/`：测试集评测；`hd_replay/wr_cmp.txt`：整场回放对比。
- `real_sessions/capture_ab.py`、`capture_noise.py`：真人逐次 A/B 与噪声基准，`capture_ab/summary.json`、`noise_base2.json` 为汇总（逐条转写与原始结果只留在服务器）。
- 回退：把 `configs/demo_chat.yaml` 里 `stop_only` 定义后的暂离句删掉，重启 demo。
