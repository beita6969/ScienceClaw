# FoR49 solver / formal closeout (2026-10-04)

本轮只核查现有代码、formal trajectory 和已有报告，没有重复任何已消耗的 ID/OOD item，也没有读取 hidden label 或改 scorer/acceptance。

## 结论

当前 solver 路由没有发现新的安全分数修复。提交 `b52d030` 已覆盖本轮暴露的唯一可修复问题：FoR49 objective 现在明确要求困难 QF_NIA 优先使用 `query_timeout_s=120`，并要求 timeout retry 的**最新** `z3_check.status` 直接连到 `submit.y` 后结束；禁止在 code 节点重写 solver 或让 retry 节点与最终提交断开。

代码侧 `SmtAdapter.z3_statuses` 将 timeout、memory、workers 和参数限制在文档范围内，并拒绝非法参数；`Z3Runner` 对 CLI error、timeout、memout 和错误输出 fail-closed 为 `unknown`。本地 `tests/test_task_FoR49.py` **12 项全部通过**，已有远端源 SHA/编译核验通过。因此没有理由改动 solver、验收或 scorer。

## 已有正式证据边界

- `SOTA49B`：ID 4/4，primary `0.875/0.875/0.9375/0.9375`，均值 `0.90625`；OOD 4 项中 3 项接受，失败项为正常 `z=0`（3 个 unknown），不重跑。
- `SOTA49C`：仅为互斥 OOD-only 批次；OOD-12/13/15 完成且接受，primary `0.875/0.9375/0.8125`，均值 `0.8750`；OOD-14 只产生 work trajectory，未有 result/scorer receipt，step 被显式取消（非 OOM），按每项只评一次规则不重跑。
- 每个已计分 episode 的最终图均经 validator 核验为 `z3_check.status -> submit.y`；以上批次仍分别是不完整批次，不能合并宣称完整 paired SOTA49。
- FoR49 deterministic ID 池只有 `00–11` 且已耗尽；剩余 OOD `16–27` 没有未消耗的 ID 对应池，不能组成新的完整 ID/OOD formal 批次。剩余 OOD 只能在有明确需要时做单独、明确标记的 OOD-only 证据。

因此 FoR49 当前应收口为“solver 路由与可审计性已修复；已有分数高且正式证据部分完成；无安全新代码或可补齐的 paired formal item”。
