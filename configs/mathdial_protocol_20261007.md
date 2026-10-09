# E1 MathDial 双协议复制实验：冻结协议

冻结时间：2026-10-07（全量调用之前）。用途：把论文从 QATD2k 单数据集扩展到第二个数据集，支撑 ARR long paper。

## 设计

- 数据：dialogue-kt 仓库自带的 MathDial `mathdial_{train,test}_atc.csv`（2,253 / 595 段，13,356 / 3,307 学生轮）。许可：CC BY-SA 4.0（dialogue-kt README 原文）。轮次编号与 KT 管线完全一致。
- 标注模型：`gpt-6.1-sol`，用户指定网关；两协议同一模型、同一时段跑。与 QATD2k 主标注器（gpt-5.6-sol）不同——**跨数据集比较同时改变了数据集和标注器版本，论文必须写明**。
- Full：LLMKT `ANNO_CORRECTNESS_SYSTEM_PROMPT` 原文 + LLMKT 原 MathDial 描述，整段一次调用。
- Prefix：QATD2k `annotate.SYSTEM_PROMPT` 的规则与示例不变，仅把任务描述换成 MathDial（题目 + 正确解 + 学生初始错误解）；逐轮前缀递增，保留之前的 assistant 输出（与 QATD2k 主协议一致）。
- 上下文：两协议都看到题目、正确解与学生初始错误解（LLMKT 原设定）。
- KC：沿用 LLMKT 原 MathDial ATC 标准 KC（gpt-4o 产，149 个），两协议共用，**不随协议变化**；协议只改 correctness。某轮 KC 为空则 dialogue-kt 视为 NA，两协议同样处理。
- 末轮：dialogue-kt 对 MathDial 末轮用学生自报 `self_correctness` 覆盖 correctness（若该轮非 NA）。两协议同样适用，论文需说明。
- 重试：瞬时错误最多 1 次重试；解析失败最多再请求 1 次；Prefix 某轮解析失败则该对话停止（不续接未解析历史），对话整体排除并报告数量。
- 调用上限：test full 700、test prefix 4,000、train full 2,600、train prefix 16,000（含重试）。

## 预先写死的判定（结果出来后不改）

1. **H1 scoreability 主导**：两协议分歧中 NA 边界变化占比 > 50%。
2. **H2 方向**：Prefix 非 NA 而 Full 为 NA 的轮数多于反向（报计数与对话聚类 bootstrap CI）。
3. **H3 polarity**：双方非 NA 时 correctness 一致率（只报告，不设阈值）。
4. **下游**：DKT / DKT-Sem / simpleKT / LLMKT 的 Train×Eval 2×2，共同预测支持集 AUC + paired Δ（与 QATD2k 同方法）。

任一假设不成立都如实写入论文；结论将变为“协议效应依赖语料/标注器”。

## 试点（不进入正式结果）

`audit_materials/mathdial_pilot_20261007/`：test 前 3 段，Full 3 次调用、Prefix 17 次；17 轮中 6 轮分歧（5 个 NA 边界，Prefix 非 NA/Full NA 方向占 4 个），0 解析失败。正式运行从头独立执行，不复用试点标签。

## 冻结哈希

```
2629bda61454f7eb6ab6f84be755144416f0cc998d01fd508af1a200904e7de8  causal_annotation/annotate_mathdial.py
5b0a83aa34c4c1487b81c1bdce5fe4ca70e27e69fc7f1ab1df819b6edb32788e  causal_annotation/annotate.py
bafec923c75c7ab482f17d627deb6324076737da6543e3dc1973cab5f95b76c7  causal_annotation/annotate_leakage.py
e60158125713f0df2bdc573c35bfc12393227dce76d7f990742673c4291b3f4b  dialogue-kt/data/annotated/mathdial_train_atc.csv
2c53bb668346599b9308d0bc03760eeefd3af0f78d66e060c94af4218b22c1c1  dialogue-kt/data/annotated/mathdial_test_atc.csv
```

正式输出目录：`audit_materials/mathdial_formal_20261007/{full,prefix}_{test,train}/`。
