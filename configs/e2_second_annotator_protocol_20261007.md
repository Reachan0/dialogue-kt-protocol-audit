# E2 第二标注器（QATD2k 测试集双协议）：冻结协议

冻结时间：2026-10-07（全量调用之前）。用户已同意使用 deepseek-v4.1-flash，并同意将 QATD2k 测试集发送给该模型。

## 目的

检验 QATD2k 主发现（分歧以 scoreability/NA 边界为主；Prefix 比 Full 更常给出可评分标签）是否换一个不同家族的标注器后仍成立。只做标签层分析，不训练 KT 模型。

## 设计

- 数据：QATD2k 测试集全部 395 段、4,404 学生轮（`data/dialogues_test.jsonl`）。
- 模型：`deepseek-v4.1-flash`，经用户指定网关；Prefix 与 Full 同一模型、同一时段运行。
- Prompt、请求构造与解析：**原样调用** `annotate.annotate_dialogue`（Prefix）与 `annotate_leakage.annotate_dialogue`（Full），与主标注器完全一致；只替换客户端（预算、重试、模型名校验、响应台账）。
- thinking：请求已传 `thinking: disabled`，但试点中该模型仍返回 thinking 块。两种标注器都只用 text 块解析标签并构造历史，thinking 内容不回放；台账记录每次响应的块类型和 token 用量。论文需说明该模型存在不可关闭的内部推理。
- 调用上限：Prefix 6,000、Full 600（含重试与解析重试）。
- 返回模型名一致性只说明网关的报告，不证明上游真实模型或版本。

## 预先写死的分析（结果出来后不改）

与主标注器相同的口径：
1. **H1**：E2 自身 Prefix/Full 分歧中 NA 边界占比 > 50%。
2. **H2**：E2 自身“Prefix 非 NA、Full 为 NA”多于反向（计数 + 对话聚类 bootstrap CI，2,000 次）。
3. **H3**：双方非 NA 时 correctness 一致率（只报告）。
4. 跨标注器：E2 与主标注器在同一协议内的一致率（Prefix–Prefix、Full–Full），说明标注器差异与协议差异的相对大小。

任一假设不成立都如实写入论文，结论收窄为“协议效应依赖标注器”。

## 试点（不进入正式结果）

`audit_materials/e2_pilot_20261007/`：测试集前 3 段，24 轮；Full 3 次调用、Prefix 24 次；0 解析失败；E2 自身 Prefix/Full 分歧 1/24；与主标注器同协议一致 22/24（两协议各自）。所有响应块均为 [thinking, text]，stop 均为 end_turn。正式运行从头独立执行。

## 冻结哈希

```
847e1c739f9df44217636b13bdac328a8f2fea0bf56d602b880aa541d4597c40  causal_annotation/annotate_second_annotator.py
5b0a83aa34c4c1487b81c1bdce5fe4ca70e27e69fc7f1ab1df819b6edb32788e  causal_annotation/annotate.py
bafec923c75c7ab482f17d627deb6324076737da6543e3dc1973cab5f95b76c7  causal_annotation/annotate_leakage.py
2629bda61454f7eb6ab6f84be755144416f0cc998d01fd508af1a200904e7de8  causal_annotation/annotate_mathdial.py
d940869f8639b0c1f63a9c91b81da1495effd4bb5b7b894b30698dd72c4b9ecc  data/dialogues_test.jsonl
96ac2e3db075130dd09dc64d648610629063e553073d9528f791c7d84e55dfbb  data/causal_labels_test.jsonl
b8cbe5fcaa4ba90618775e66365d0df320589c9d2cb8a11858835791b2086528  data/leakage_labels_test.jsonl
```

正式输出目录：`audit_materials/e2_second_annotator_20261007/{prefix,full}_test/`。
