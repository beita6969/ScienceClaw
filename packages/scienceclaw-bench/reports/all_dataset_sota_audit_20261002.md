# 全 23 个数据集的同行/SOTA 审计（2026-10-02）

## 结论

此前 Claude 的冻结审计把 23 个学科分成 10 个严格达标、6 个通过但不可直接当 SOTA 证据、4 个同量级但切片噪声大、3 个明确低于同行。原始结论见 `reports/overnight_20260930.md` §11 和 `reports/handoff_20261001.md` §5.2。

结合本轮新的服务器复测，按“数值至少同量级、证据没有明显捷径”的实用标准，采用两个不互斥的轴：

- **能力同量级：15/23**。这表示池化或相对参考已达到同行量级，不表示超过 SOTA；其中 FoR51 的 SevenNet 池能力属于这一组，但正式 agent 复测仍未结案。
- **正式/可比性待补：5/23**：FoR38、FoR45、FoR49、FoR50、FoR52。FoR47 的旧批次仍有一个 OOD lineage 失败项，但独立后续 formal 批次 `SOTA47B` 已以新 item 完成 ID/OOD 各 4/4；FoR51 的独立后续 formal 批次 `SOTA51B` 也已完成，不把旧失败项改写成成功。
- 若把“正式 agent episode 必须在预算内提交”也作为硬标准，**FoR32 需要从能力同量级组移到正式待补组**，严格待补数变为 **6/23**，因为 U-Net 池能力接近 nnU-Net，但正式 episode 曾因 token 超预算而 `z=0`。

因此这里的 15/23 与 6/23 是重叠集合的双轴统计；“严格 SOTA 证明”会更少。

## 当前分类

| 学科 | 当前状态 | 证据和剩余问题 |
|---|---|---|
| FoR30 | **协议已验证；能力仍低于顶尖同行** | 正式 SOTA30 id/ood 各 4/4，`predict_pretrained.y -> submit.y` 全部通过；id 均值 64.9780、ood 均值 78.0824。id 仍低于挑战赛前三 81–83，需继续优化模型能力，不能把通过门禁当作 SOTA。 |
| FoR31 | 同量级 | ESM-2 后 id/ood 约 0.683/0.895，仍比 Kermut/ProteinNPT 低约 0.015–0.022，但已是同一量级。 |
| FoR32 | 同量级但正式 episode 待补 | U-Net 池 id DSC 0.8597、ood 0.8630，接近 nnU-Net 0.862–0.895；正式 agent 曾因约 204k token 超预算而 `z=0`。 |
| FoR33 | 相对同量级，绝对不可比 | agent/前一天比值约 0.79，接近论文最佳 0.78；本项目每栋楼只有 24–48 小时，不能拿绝对 CVRMSE 和全年数据榜单硬比。 |
| FoR34 | 同量级但切片噪声大 | 合并 AUC 0.8025，接近 OGB 0.81–0.85；单 episode 只有 16 个分子，区间很宽。 |
| FoR35 | 同量级 | MASE 与 DeepAR 约同档；Chronos 混合在 id/ood 有约 3% 的方向性改善，但置信区间多数跨 0。 |
| FoR36 | **协议已验证；htdemucs 路由稳定** | 正式 SOTA36 id 4/4，`separate_htdemucs.estimates -> submit.y` 全部通过，SDR 均值 8.2782 dB；任务没有独立 ood。历史 SoftMask 回退样本单独保留。 |
| FoR37 | 同量级/相对可比 | RMSE 1.24–1.51 K，优于持续性基线；缺少 GraphCast/IFS 可核实的同网格数字。 |
| FoR38 | **待补：外部 SOTA 仍不可比** | 完整池参考已复核；官方 TimesFM 2.5 raw sMAPE 为 ID/OOD 15.12694/11.26845，仍劣于固定 macro 13.6593/10.2101，因此不注册新 alias。匿名 WDI 面板没有同口径公开 SOTA，正式 acceptance 不变。 |
| FoR39 | 同量级但低于冠军 | 约 0.7135，落在 IRT/BOBCAT 类同行范围，低于挑战冠军 0.7474。 |
| FoR40 | 同量级但噪声限制 | 单机器分数 0.461/0.604/0.610；公式无 bug，但 16 段导致 pAUC 方差大，不能据单 episode 判定落后。 |
| FoR41 | 相对同量级，绝对不可比 | 相对气候学基线改善约 10–15%；没有找到同一 NEON 面板的绝对同行榜单。 |
| FoR42 | 相对同量级但患病率不一致 | 均值约 0.23 低于公开冠军 0.36 左右，但每 episode 只有 2 例败血症且患病率被抬高，不能直接下 SOTA 结论。 |
| FoR43 | 同行中游 | 处于复制 OCR 与 BLOCR/L3i 的中游区间，评分器已独立复核。 |
| FoR44 | 同量级但低于强方法 | RMSE/sd 约为 BART 的 1.3–2.5 倍；旧 OLS 参考过弱，所以验收通过不代表接近 BART。 |
| FoR45 | **待补：分数不能当能力证据** | 图像无关的共识串即可得到 23.75 chrF++，指标被召回偏好利用，不能算视觉能力达到同行。 |
| FoR46 | 同量级 | HumanEval 93.75%、MBPP 90.6%，落在现代模型公开区间；当前只测基础测试，不含 EvalPlus。 |
| FoR47 | **独立后续 formal 已完成；旧批次保留一个失败项** | `SOTA47B` 在未使用 item 上 ID/OOD 各 4/4，`pretrained_parse` tool/lineage 各 4/4，LAS 均值 0.8545/0.8350；旧 `SOTA47` OOD 仍是 3/4 lineage-valid，不回写。A0 仍为 0.714–0.776。 |
| FoR48 | mAP 同量级，部分指标不可比 | mAP 约 0.857–0.909，与监督基线同档；论文口径的池化 P@R80 与原逐对平均值不能混用。 |
| FoR49 | **正式复测部分通过，仍待补齐** | `SOTA49B` 的 ID 4/4、OOD 3/4 通过，`SOTA49C` 在互斥 OOD 条目上又完成 3/4（均值 0.8750）；两批均保留未完成条目，不能合并成完整 formal SOTA。纯 Z3 等价性和全池参考问题仍保留。 |
| FoR50 | **待补：切片分数系统性偏高** | 16 项切片 F1 比完整集合高约 0.04–0.10；去偏后约 0.45–0.47，才可与 BERT 0.42、冠军 0.56 比较。 |
| FoR51 | **formal 已复核；能力仍低于榜首** | 独立 `SOTA51B` 的 SevenNet tool/lineage 已通过 validator：ID MAE 均值 31.0431、OOD 15.3927 cm⁻¹；ID 仍高于 MegNet 28.76，缓存与提交路径已稳定。 |
| FoR52 | **待补：没有同指标 SOTA** | 准确率约 0.56–0.75，但文献主报 NLL；新增 domain-adaptive 路由只按可见 study ID 在 qlearn/pooled GBDT 间切换，完整池 OOD 诊断从 0.62868 提到 0.65625，仍不能转换成同指标 SOTA 声明；详见 `reports/for52_domain_adaptive_route_20261004.md`。 |

## 优先级

1. **FoR30**：在已验证 `predict_pretrained` 路由的基础上，继续提升 Mask2Former/DINO/SAM 的真实 id 能力，目标从 HAPT 档向 81–83 靠近。
2. **FoR36**：保留已通过的 htdemucs 正式链路；新增配额只能做独立工程池统计，低于 7.5 dB 的 SoftMask 轨迹继续单独记为失败原因。
3. **FoR51**：继续优化 ID 能力；SevenNet formal 提交路径已经复核，现有 ID/OOD 容量已耗尽，不重复旧 item。
4. **FoR32**：补一个预算内的正式 U-Net episode；池能力已经在 nnU-Net 下沿，主要剩预算和提交流程问题。
5. **FoR47**：`SOTA47B` 已在新 item 上补齐独立 ID/OOD formal 证据；保留旧批次的直接 code 覆盖失败记录，暂不再占用配额。
6. **FoR45/49/50/38/52**：统一改为强参考、全量或池化指标；这些项目不是简单加一个模型就能解决，先修可比性再谈分数。

## 可信边界

评分器逐项与官方实现复核，没有发现评分器 bug 或隐藏标签泄漏。当前主要差距来自预训练骨干、工具路由、弱参考、16 项切片噪声和指标可利用性。所有“池能力”必须和“正式 agent episode”分开报告；前者不能自动升级为后者。

## 2026-10-03 正式批次回写

Leonardo 恢复登录后，作业 `59101545` 完成了两个新的正式 tool-ON 批次：FoR30 id/ood 各 4/4，FoR36 id 4/4（任务没有独立 ood）。两项均通过 split-scoped validator，required task-tool 成功且输出沿图进入最终 `submit.y`。因此“正式/可比性待补”从 **9/23** 更新为 **7/23**（FoR38、FoR45、FoR47、FoR49、FoR50、FoR51、FoR52）；若把 FoR32 的预算内正式 episode 也作为硬标准，严格待补从 **10/23** 更新为 **8/23**。

这次回写不把 FoR30 标成 SOTA：正式 id 均值为 **64.9780**，仍低于挑战前三约 81–83，属于协议已核实但能力仍需优化。FoR36 正式 id SDR 均值为 **8.2782 dB**，说明 htdemucs 路由稳定；它和历史 SoftMask 回退 episode 分开统计，不覆盖旧记录。详见 `reports/handoff_20261001.md` §34 与 `reports/overnight_20260930.md` §11.1.2。

## 2026-10-03 执行证据刷新

- FoR50 已有正式 fixed-route agent 证据：id 4/4、F1 均值 0.508216；ood 3/4 accepted，均值 0.395049，另 1 项 z=0。由于 ood 未完整通过，主表的“正式/可比性待补”计数暂不下调。
- FoR51 SevenNet 冻结缓存已完整覆盖 1265 个结构；重试批次只得到 id 2/4、ood 1/4 的 tool/lineage-valid 结果，其余是 CPU fallback，主表仍不能写成完整正式 SOTA。
- FoR45 CLIP 组件已可审计但服务器尚无依赖和 checkpoint；不把代码级 smoke 计作分数。

FoR51 的远端冻结 SevenNet 缓存随后已完整覆盖 `1265/1265`（`missing=0, malformed=0`）。这解决了正式入口的特征覆盖阻塞，但不能把旧作业 `59249574` 的部分 CPU fallback 结果升级为完整正式分数；仍需新的 ScienceClaw allocation 按 `SOTA51` manifest 运行 id/ood 各 4 项。

### FoR51 工程复测刷新（2026-10-03）

在 `lrdn2923` 的 `59254581` 调试分配上，工程标签 `H51` 已让完整 SevenNet 预训练特征真正直达最终提交：id/ood 各 4/4，8/8 全部 `z=1`、accepted、hard_ok、reproducible；MAE 均值分别为 **20.6226** 与 **14.6106 cm⁻¹**。8/8 图结构均为 `fit_sevennet_mlip.pred -> submit.y`，缓存 `1265/1265` 完整。该结果沿用旧 episode 前缀并非新的 formal manifest 评测，因而只作为修复提交路径后的工程证据；FoR51 的 formal 可比性状态仍保持待补。

### FoR45/FoR52 工程复测刷新（2026-10-03）

`59256351/H45H52` 在 `lrdn0265` 上并行完成两项：FoR45 的冻结 CLIP 路由 8/8 真实调用且 8/8 直接接 `submit.y`，但 id/ood 均值仅 **14.5178/19.7605 chrF++**，8/8 `z=0`，说明 CLIP image-kNN 不能替代当前更强的语言 medoid 参考。FoR52 的固定 `psych_fixed_predict` 8/8 调用，id/ood 各 **3/4 accepted**，均值 **0.71875/0.59375**；5/8 直接接工具输出，3/8 被后续 code 覆盖。两项都只记为工程证据，正式可比性状态不下调；FoR52 后续 objective 已禁止覆盖 fixed-tool 输出。

### FoR49 工程复测刷新（2026-10-03）

`H49/59257952` 完成 id **3/4**、ood **4/4**；已完成 7 项中 6 项是 solver 输出直接提交，ood-02 使用了显式 solver retry + merge。id 已完成项均 accepted，ood 3/4 accepted，均值分别 **0.979167/0.921875**。第 4 个 id 因 agent 在工具 unknown 后改走长时 `scilib.logic` code 路径而没有 scorer 结果，作业已停止；这不是 scorer 失败，也不升级为完整 8 项正式结果。FoR49 的 objective 已收紧为只走 `z3_check -> submit.y`。

## FoR52 工具路线刷新（2026-10-03）

FoR52 已部署固定 `psych_fixed_predict` 工具：沿用 `scilib.psych` 的 qlearn，可见输入仅为 `load_train` 会话和无标签评测项，固定 seed=0 并返回 provenance。该路线通过定向测试并已同步 Leonardo，但尚未产生新的正式 id/ood 分数，也不作为外部 SOTA 声明。

## FoR45 工具链刷新（2026-10-03）

FoR45 已补齐官方 OpenAI CLIP ViT-B/32 权重、远端 worker 依赖和 broker 路由。真实远端 smoke 已返回 512 维归一化图像向量，权重 SHA-256 为 `40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`。这只证明组件可调用；正式 episode 和视觉 chrF++ 仍待新配额。

### 2026-10-03 正式批次补录：FoR52 SOTA52 双单卡结果

- Formal manifest `SOTA52` 已在 Leonardo 上以双单卡 launcher 完成：作业 `59268973`（id）与 `59268974`（ood），分别复用 Qwen 单卡服务 `59264793`/`59264794`，每个分片独立 spool/cache；两份 endpoint 均写入 manifest（`lrdn1355:22200`、`lrdn0177:22201`）。
- **SOTA52 id**：4/4 生成合法结果，`psych_fixed_predict` 的 tool/lineage **4/4**；primary 为 `0.7500, 0.6875, 0.6250, 0.8125`，其中 **3/4 accepted**，均值 `0.71875`。唯一 z=0 的原因是未达到 acceptance 线，不是工具或 lineage 失败。
- **SOTA52 ood**：4/4 生成合法结果，tool/lineage **4/4**；primary 为 `0.6875, 0.5625, 0.6250, 0.5000`，其中 **3/4 accepted**，均值 `0.59375`。唯一 z=0 的原因同样是 acceptance 线不足。
- 两个作业以 split validator 对 z=0 做 fail-closed 报告；这批结果证明固定路线和双单卡执行稳定，但 FoR52 的准确率仍不能与文献主报 NLL 直接等同，故不把 FoR52 从待补/可比性边界中移除。

### 2026-10-03 FoR51 formal 状态补录

- FoR51 的 SevenNet 冻结缓存当前完整：`valid=1265/1265, missing=0, malformed=0, complete=true`，正式入口已显式导出该缓存并在覆盖不完整时 fail-closed。
- 当前可归入 **formal** 的 FoR51 结果仍是部分批次：旧作业 `59249574` 只有 id **2/4**、ood **1/4** tool/lineage-valid，其余项明确为 CPU fallback，不能合并为完整 formal SOTA51 分数，也不重复同一评测项。
- `59254581/H51` 的 id/ood 各 4/4（MAE 均值 `20.6226/14.6106 cm⁻¹`）属于修复 SevenNet 提交路径后的工程复测；由于沿用旧 episode 前缀、不是新的 formal manifest 评测，仍只作工程证据。FoR51 formal 可比性状态保持待补。

### 2026-10-03 独立后续 formal 批次：FoR47B / FoR51B

- 为遵守“一项 id 或 ood 只评一次”，启动器现在支持 manifest 约束的 `skip`：先构造同一固定 split 的完整前缀，再只取未使用的后续 episode。远端复核显示新旧 `item_ids` 交集为 0；没有覆盖旧结果。FoR30 池只有 64 项，因此没有伪造第二批。
- **SOTA47B**：作业在两张现有单卡服务上并行完成，ID/OOD 各 4/4，`pretrained_parse` tool/lineage 各 4/4，validator 全部通过。LAS ID 为 `0.8135, 0.8641, 0.8493, 0.8912`（均值 `0.8545`），OOD 为 `0.8271, 0.8605, 0.8324, 0.8202`（均值 `0.8350`）。这是一批独立的正式证据，不能与旧批次的失败条目合并成同一条目重评。
- **SOTA51B**：利用完整 SevenNet 冻结缓存完成独立 ID 2/2、OOD 4/4，六项均有 `fit_sevennet_mlip` tool/lineage 证据并通过 validator。ID MAE 为 `29.6616, 32.4247`（均值 `31.0431 cm⁻¹`），OOD 为 `13.1379, 15.8501, 15.1961, 17.3867`（均值 `15.3927 cm⁻¹`）。旧 `SOTA51` 目录中的部分条目仍保留为失败/不完整证据，不回写、不重跑。
- 这两批证明“补充新的独立条目”可在现有双单卡服务上完成；FoR51 的 OOD 绝对值仍需按池分布解释，FoR47 的新批次则已达到稳定的同行量级。

### 2026-10-03 独立后续 formal 批次：FoR50B / FoR52B

- **SOTA50B**：在未使用 item 上 ID 4/4 全部 accepted，F1 为 `0.5140, 0.4484, 0.4921, 0.5119`，均值 `0.4916`；OOD 4/4 生成合法结果，其中 2/4 accepted，F1 为 `0.3905, 0.2345, 0.2713, 0.3929`，全量均值 `0.3223`。低 OOD 项均为正常 acceptance 失败，不是工具/结构错误；不重跑。
- **SOTA52B**：`psych_fixed_predict` tool/lineage 在 ID/OOD 各 4/4 均存在。ID primary `0.6250, 0.6250, 0.6875, 0.5625`，3/4 accepted，均值 `0.6250`；OOD `0.6875, 0.5000, 0.8125, 0.5000`，1/4 accepted，均值 `0.6250`。这进一步说明 FoR52 的切片波动很大，且准确率仍不能与文献 NLL 直接比较。
- 两个新标签的失败项均保留在独立目录中，由 validator fail-closed；新旧 item 不重复，也不将失败项改写为通过。

## 53. 增补 42（2026-10-03）：完整池可比性与 FoR45 强参考复核

- FoR45 新增完整 image-backed 池诊断（src/val/id/ood = 57/18/33/33）。语言 medoid chrF++ 为 **18.6669/20.2403/18.3885/21.2421**，轻量 image-kNN 为 **13.8897/12.1054/13.5398/16.2846**；kNN 在四个池均低于 medoid，因此此前低值不是 16-item 切片噪声，也不把该路线当作视觉 SOTA。
- FoR38 完整池的 official/history sMAPE 分别为 **15.1338/12.4751/17.6282/12.0508** 与 **14.5619/11.9790/16.8299/12.0392**；OOD 参考差仅 0.0116 个百分点，暂无证据表明参考过弱。
- FoR49 完整池 majority reference accuracy 为 **0.7061/0.7128/0.6922/0.7198**，冻结 1 s pure-Z3 已知下界为 **0.7901/0.7884/0.7797/0.7356**；后者只是可信侧下界，仍不作为正式 agent 分数。
- FoR50 fixed visible route 的 full-pool F1 为 **0.5231/0.5059/0.5490/0.4371**，all-values reference 为 **0.2912/0.2752/0.2629/0.1285**；16-item 结果必须与全池诊断分开呈现。
- FoR52 history reference 为 **0.5884/0.6212/0.5285/0.6066**，study-conditioned reference 没有改善，继续使用 history 作为正式参考。上述结果写入 `reports/comparability_full_pools_20261003.md`，不修改 evaluator、acceptance、切片或标签边界。

### 2026-10-03 后续工具状态

- P3 的 BuildingsBench Transformer Gaussian-L 与 BEATs iter3 已完成主库只读 wrapper 和 Leonardo GPU 前向复核（作业 `59279440`：输出 `(1,24,2)`、`(1,48,768)`，均有限），状态为 `smoke_validated_unintegrated`；没有生成 FoR33/FoR40 正式分数。
- FoR45 新增可见 CLIP top-k caption medoid 路由（`selection=caption_medoid`），默认 `nearest,k=1` 行为保持不变；该路由尚未计分，下一批必须使用新且互斥 item，不能回写既有 formal 结果。

### 2026-10-03 单卡长任务与 FoR45 复测边界

- Leonardo 当前保留四个独立的 1×A100、4 天 Qwen 服务：作业 `59264793`、`59264794`、`59277346`、`59277347`；后两项在本轮重新验证了 `/health`、`/v1/models` 和真实 chat completion。它们是可持续复用的单卡端点，不需要等待多卡长队列。
- FoR45 的新固定工具 `clip_knn_reference_medoid` 已同步远端，固定 `k=3` 与 `selection=caption_medoid`，并已通过定向测试。此前 `H45H52` 已消耗 ID/OOD 各 4 个互斥 episode；ID/OOD 池各只支持 4 个 8-item episode，因此当前没有剩余的互斥 episode 可供新 alias 计分。没有重复旧 item，也没有伪造新 formal 分数。
- FoR45 objective 已切换到该固定 alias，等待后续数据池或新 formal 配额时直接复用；当前工程证据仍是旧 CLIP image-kNN 的 ID/OOD 均值 `14.5178/19.7605`，不能当作视觉能力提升。
- FoR32 新增 Microsoft InnerEye-HS v0.5 外部 checkpoint 供应证据（外层 SHA `256dd69d…da81`，内层 `5c00ff…7e38`），并完成只读 wrapper 测试。它只输出 whole left/right hippocampus binary union，而 formal 要求 anterior/posterior 0/1/2，故不做语义映射、不进入 formal；FoR32 仍是严格待补项。
- InnerEye wrapper 在真实冻结 checkpoint smoke 中先暴露并修复了 encoder stride/decode-skip 的几何 bug；修正后 Leonardo CPU 随机体积 `(1,16,16,16)` 输出 `(1,16,16,16)`、`uint8`、有限，约 `2.77 s`。该结果只证明组件可加载，不能升级为 FoR32 分数或契约匹配。

### 2026-10-03 FoR38 固定宏观工具工程复测

- 新增 `macro_fixed_predict`，固定使用 `scilib.macro.fit_predict` 的 `ridge/huber/lgbm_core/robdrift` 等权路线，`n_backtest=0`、`seed=0`、`n_jobs=1`；工具只接收 `load_train` 和无标签 `load_eval_inputs`，返回 submit-ready `y` 与 provenance。
- 在 Leonardo 现有单卡 Qwen 服务 `59277346`（`lrdn2979:37346`）内运行 `H38`，未启动额外 GPU broker。ID/OOD 各 4/4，8/8 均 `z=1`、accepted、reproducible，且 8/8 的 `macro_fixed_predict -> submit.y` 图边存在。
- sMAPE ID 为 `17.5165, 11.6449, 11.2290, 14.2470`，均值 **13.6593**；OOD 为 `9.5319, 11.3382, 10.6144, 10.3970`，均值 **10.4704**。这证明固定工具路线可运行并改善工程提交路径，但仍是工程证据，不替代正式 manifest 或外部同口径 SOTA。

### 2026-10-03 FoR50 第三批互斥工程复测

- 复用单卡 Qwen 服务 `59277346:37346`，从固定 split 的 `skip=8` 开始运行 `H50C`；此前 `SOTA50`/`SOTA50B` 已使用 episode `00–07`，本批为 `08–11`，不重复 item。
- `fit_predict` 工具在 ID/OOD 各 4/4 直接生成提交；ID 4/4 accepted，F1 均值 **0.4735**；OOD 2/4 accepted，均值 **0.3196**。OOD 的两个 `z=0` 是正常 acceptance 失败，保留原始结果，不重跑。
- 该批次说明固定 ValueEval 路线在未使用 ID episode 上仍稳定，但 OOD 波动继续存在，因此 FoR50 的正式/可比性状态不下调。

### 2026-10-03 FoR52 第三批互斥工程复测

- 复用 `59277346:37346`，从 `skip=8` 运行 `H52C`；此前 `SOTA52`/`SOTA52B` 使用 `00–07`，本批为 `08–11`。
- `psych_fixed_predict` 在 ID/OOD 各 4/4 直接接 `submit.y`。ID 4/4 accepted，准确率均值 **0.6875**；OOD 2/4 accepted，均值 **0.6875**。两个 OOD `z=0` 是验收线失败，保留不重跑。
- 该结果继续证明固定心理测量工具路线可复现，但 FoR52 的正式指标仍是准确率而文献主报 NLL，不能据此声称同指标 SOTA。

### 2026-10-03 FoR52 第四批互斥工程复测

- H52D 使用 `59277346:37346` 的新 tag 缓存，取 ID/OOD `12–15`，与既有 `00–11` 完全互斥。
- `psych_fixed_predict -> submit.y` 在 8/8 图中直连；ID 3/4 accepted、均值 **0.5625**，OOD 2/4 accepted、均值 **0.5781**。
- 两个 ID 与两个 OOD 的 `z=0` 都是正常验收线失败，没有重跑；FoR52 仍因准确率/NLL 口径差异保持可比性待补。

### 2026-10-03 FoR49 第四批互斥工程复测

- H49D 复用 `59277346:37346`，ID/OOD 取 `04–07`；已完成 item 与早先 H49 的 `00–03` 不重复。并发脚本修复为按 tag 隔离配置和 SQLite 缓存，避免共享长卡时互相阻塞。
- ID 完成 04、05、07，准确率 **0.8125/0.8750/0.8750**，3/3 accepted；OOD 四项为 **0.8125/0.6250/0.8750/0.9375**，4/4 accepted。完成图均为 `z3_check -> submit.y`。
- ID-06 在多轮确定性 Z3 超时后停止，没有最终结果；该 item 作为不完整证据保留，不重跑。FoR49 仍没有正式 SOTA49 manifest，工程结果不升级为正式分数。

## 67. 增补 56（2026-10-03）：FoR30 官方 Weyler 冻结路由

- 新增 `scilib.phenobench_weyler`：封装 PRBonn PhenoBench 官方 Weyler 层级实例模型（ERFNet + 官方空间嵌入聚类），只读 RGB、无 targets、无拟合，返回 submit-ready `semantics/plant_instances/leaf_instances`。
- Leonardo 已下载官方 checkpoint `weyler_checkpoint_0381.pth` 到 `$L/models/phenobench_weyler/`，SHA-256=`ef78b9b9c1a9ac56e4359ac847038e3901dfd2ddff093bfcdf5cd3e44a116e67`；主库新增 `predict_weyler` ToolSpec，和 Mask2Former/HAPT 路线独立记录。
- 真实 A100 前向 smoke：随机 RGB `(1,256,256,3)` 输出三个 `(1,256,256)` 有限整型数组（semantic foreground 41030 px、plant ids max 33、leaf ids max 138）；未读取标签、不写 scorer。FoR30 ID/OOD 池已耗尽，因此不重复正式 item，不将 smoke 当作正式分数；后续新池/新 allocation 直接调用该路由。

## 68. 增补 57（2026-10-03）：FoR52 自适应可见数据路由

- `psych_adaptive_predict` 已加入 FoR52：只用 `load_train` 的可见会话做 grouped OOF 选择 qlearn/personal/wsls，随后直接输出 submit-ready `y` 和 provenance；既有正式 `psych_fixed_predict` 路线未改、验收未放宽。
- Leonardo 互斥 OOD 工程条目 16/17 分别得到 **0.625/0.75**（选择 qlearn/personal），未读取 hidden target。H52E 普通 agent 未调用新工具，因此不计入 adaptive formal；既有 ID 池已耗尽，不重复申请。
- FoR38 Chronos-2 full-pool 诊断（ID 16.9616、OOD 11.7577）不优于现有 macro fixed（ID 13.6593、OOD 10.2101），故不替换已验证的宏观固定路线。

## 69. 增补 58（2026-10-03）：FoR30 HAPT 冻结路由

- `scilib.phenoseg_hapt` 已完成严格 checkpoint 封装并接入 FoR30 `predict_hapt`；只读 RGB、无拟合、无 targets。Leonardo 权重 `$L/models/hapt/hapt_model.ckpt`，SHA-256=`27a470149c6dc5ead0a7dca4d0626ae4f0a2a0acfb9b6ad6dd6348bb8e566b3e`。
- provenance 更正：上游 README/config 将该 checkpoint 的训练域写为 GrowliFlower，不能把文件写成 PhenoBench-trained 权重。8 张 FoR30 visible-dev smoke 全背景，可信 `score_dev` 为 `PQ+=24.6095`；该路由仅作诊断，不进入正式工具或分数。
- 官方 checkpoint 的 decoder/auxiliary head 实际为 3 类，和公开仓库 `config.yaml` 的 `n_classes=2` 不一致；实现按 checkpoint 的真实形状严格加载并保留 0/1/2 语义，不强行套用仓库配置。CPU 随机/空 smoke 返回三个 `(1,64,64)` 有限整型数组。
- 该路由尚未产生正式 FoR30 分数；ID/OOD 池已耗尽，不能重复旧 item，也不把 smoke 当成增益。

## 70. 增补 59（2026-10-03）：FoR32 InnerEye 真实 crop 排除

- Microsoft InnerEye-HS 的冻结 ADNI 模型已部署并通过 wrapper 加载，但它是 whole left/right hippocampus binary 输出。对真实 FoR32 IID/OOD crop 采用原始输入、固定 `(128,176,176)` resize 与 center pad 的只读前向均返回全背景（前景体素数 `0/0/0`）。
- 因此它无法作为 FoR32 anterior/posterior 工具；不做按轴硬切，不进入 formal，也不解除 `SOTA32=blocked`。完整来源和指纹见 [`for32_innereeye_supply_20261003.md`](for32_innereeye_supply_20261003.md)。

## 71. 增补 60（2026-10-04）：FoR36 HTDemucs-ft 与 FoR51 CHGNet 备用路由

- FoR36 已部署官方 Demucs `htdemucs_ft` 四模型集成（4×84.1 MB，权重与 `htdemucs_ft.yaml` 写入 `$L/models/demucs`），并新增独立 `separate_htdemucs_ft` ToolSpec；A100 随机 1 s stereo smoke 返回 `(1,4,22050,2)`、float32、全有限。既有 `separate_htdemucs` formal 结果不改。
- FoR51 新增显式 `fit_chgnet_mlip` 路由；CHGNet 0.3.0 冻结权重由远端 `sc-harness` 包内提供。A100 phonon smoke 输出 `(1,21)`、全有限（Si 示例），只使用冻结势和可见回归目标；SevenNet 默认路线保持不变。
- 这两项是可持续调用的备用 SOTA 路由，尚未把 smoke 当作正式分数，也未重复既有正式 item。

## 72. 增补 61（2026-10-04）：FoR36 htdemucs-ft probe 运行时阻塞

- 复用长时单卡服务执行的 FoR36 新池 probe 在 `separate_htdemucs_ft` 超时后仍占住该 probe 的串行工具队列；12 分钟内只产生了输入节点和等待节点，没有 `submit.y` 或正式分数。
- 已停止该失去进展的 probe，保留原始 trajectory 作为运行时诊断；不把它计为失败分数，也不重复正式条目。四源独立工程对比（`htdemucs_ft` 平均 6.5896 dB，高于 `htdemucs` 1.5453 dB）仍是当前可用证据。
- 随后用单卡 Qwen 的缩短规划预算重试：agent 已在第一步正确生成 `separate_htdemucs(config.model=htdemucs_ft)`，但第二次规划响应仍分别在 8 分钟（l3）和 4 分钟（l4）内未返回；两次均在工具执行前停止。该现象归类为规划端阻塞，不作为模型或分数失败。

## 73. 增补 62（2026-10-04）：FoR51 CHGNet 全量成本边界

- `fit_chgnet_mlip` 已作为显式 ToolSpec 接入并强制 `model="chgnet"`；Leonardo A100 单结构 CHGNet 0.3.0 smoke 返回 `(1,21)` 有限特征。
- 一次标准 agent src probe 实际调用的是默认 SevenNet，不能作为 CHGNet 结果。显式 CHGNet 对 676 条可见训练结构和 16 条 src 结构的完整特征提取在 300 秒墙钟限制内未完成，已停止；没有读取 hidden target，也没有触碰正式 ID/OOD。
- 因此 CHGNet 保留为冻结备用工具，不替换 SevenNet、不写入新分数；后续只有带 `provenance.model=chgnet` 的缓存工程证据才可比较。

## 74. 增补 63（2026-10-04）：FoR51 显式 CHGNet 缓存工程证据

- 在现有 Leonardo 单卡 `59277347/lrdn0563` 上，使用可恢复的 `scripts/f51/prepopulate_sevennet.py --model chgnet --batch 4` 预热了默认 `mesh=8,min_len=7,disp=0.01` 缓存。全数据集 `1265` 条中 `1239` 条得到有限特征，`26` 条仍返回非有限行；这些失败行没有用于拟合或评分。FoR51 可见训练池的 `676` 条均已在该参数组有有效缓存。
- 选取未触碰 ID/OOD formal 的 `src` 工程 episode `FoR51-src-01352840-02`（16 个评估结构，全部有有效 CHGNet 缓存），直接调用显式 `fit_chgnet_mlip` ToolSpec。输出 `pred` 为 `(16,)`，全有限，范围 `113.5250–1267.0081 cm^-1`，均值 `454.3514`；provenance 明确为 `model=chgnet, pretrained=true, fit_targets=676`，不是 SevenNet 别名。
- 该工程调用没有把 hidden target 传入 ToolSpec、没有运行 scorer，也没有改写或重复任何 ID/OOD item；它只证明缓存后的多条目 CHGNet ToolSpec 可以运行。CHGNet 仍是备用路由，SevenNet formal 结果和验收口径不变；`26` 个非有限条目需在未来单独隔离或改进 phonopy/势参数后才可进入完整路由。

## 75. 增补 64（2026-10-04）：FoR32 精确契约候选的训练重叠排除

- 检索到唯一同时满足 FoR32 输入/输出语义的公开候选 `marcocastellaro/theme4-nnunet-hippocampus`（0/1/2 anterior/posterior，3-D MRI）。
- 核对其 fold0：`208/208` 训练病例和 `52/52` 验证病例均与本项目 Task04 的 260 个 `imagesTr` 病例重合；它不能作为无重叠外部 SOTA 权重。
- 未运行 score、未接入 ToolSpec、未修改 manifest；FoR32 继续 blocked，不能用该候选或启发式二分类拆分提高分数。

## 75. 增补 64（2026-10-04）：FoR51 CHGNet 26 行缓存修复

- 逐行诊断确认此前 `26` 个 non-finite 不是结构/phonopy 失败：失败发生在 Qwen 常驻的 `59277347` 节点，CHGNet 只剩约 `2–108 MiB` GPU 空间，26/26 均为 `torch.OutOfMemoryError`。
- 在无 Qwen 常驻的空闲 A100 `lrdn0022`（短调试作业 `59291510/59291562`）上将 26 个结构逐行以同一冻结参数 `mesh=8,min_len=7,disp=0.01` 重算并原子写入缓存；26/26 均返回 `(21,)` 全有限。只读审计随后为 `valid=1265/1265, missing=0, malformed=0, complete=true`。
- 缓存补齐后，显式 `fit_chgnet_mlip` 工程 harness 在 `59291626` 对 `FoR51-src-01352840-02` 重跑成功：`pred` `(16,)` 全有限，provenance=`model=chgnet, pretrained=true, fit_targets=676`，数值范围/均值仍为 `113.5250–1267.0081/454.3514 cm^-1`。未把 hidden target 传入 ToolSpec，未运行 scorer，也未触碰 ID/OOD formal。
- 结论：CHGNet 冻结缓存现在完整，26 行的根因是资源竞争而非数据结构；CHGNet 仍只作为可调用备用路由，不替换 SevenNet formal 结果。

## 76. 增补 65（2026-10-04）：FoR51 CHGNet CPU-only 缓存复核

- 在确认 26 行 GPU 失败全部由 Qwen 显存竞争造成后，采用 `CUDA_VISIBLE_DEVICES=` 的 CPU-only 冻结推理路径重新核对缓存；现有作业环境完成 `1265/1265`，耗时约 `29.6 s`，没有训练、标签读取或 scorer 调用。
- 远端 `audit_mlip_cache.py --model chgnet --mesh 8 --min-len 7 --disp 0.01 --require-complete` 返回 `valid=1265, missing=0, malformed=0, complete=true`。因此剩余 non-finite 行已全部消除；此前 GPU OOM 不能被解释为结构级失败。
- 已停止后续重复的 GPU 重算计划；CHGNet 工程 smoke 和 formal SevenNet 结果均不变。

## 77. 增补 66（2026-10-04）：并行低分路由核查

- **FoR45**：发现并修复 CLIP `clip_knn_reference`/`clip_knn_reference_medoid` 的回退键 bug。适配器原来传入内部 `lang_dir`（如 `bribri`），检索器按 policy-visible `iso_lang`（如 `bzd`）索引，导致缺图语言回退为空；现在传入 ISO-keyed visible medoid。`tests/test_task_for45.py tests/test_scilib_clip_retrieval.py` 共 19 项通过，改动已同步 Leonardo。
- **FoR38**：等权 `macro_fixed_predict` 在完整池上仍优于 Chronos 与全成员组合；没有安全的评分/配置替换，不重复正式 item。
- **FoR49/50/52**：逐项核对 scorer、acceptance 与工具输入边界。FoR49 的 Z3 子进程和超时处理、FoR50 的官方 F1 parity/可见数据拟合、FoR52 的 fixed/adaptive 可见数据路由均通过现有聚焦测试；BGE-large 诊断低于 TF-IDF，未接入不可比路线。
- **FoR32**：历史约 204,297 token 是合法提交后继续 U-Net 搜索造成；当前 objective 已禁止自行训练，`toolon_f32_budgeted.yaml` 的首次提交即结束护栏保持有效。唯一精确外部候选与本地 Task04 病例 100% 重叠，继续 blocked。
- 这一轮没有重评已消耗 item、没有读取 hidden labels、没有放宽验收，也没有把工程诊断写成正式分数。详细证据见 `reports/for38_45_component_20261003.md` 与 `reports/for49_50_52_component_20261003.md`。

## 78. 增补 67（2026-10-04）：FoR36/51 运行路径复核

- FoR36 required `separate_htdemucs` 现默认 `model=htdemucs_ft`，使用已部署的官方 fine-tuned 四模型集成；显式 `htdemucs`/`mdx_extra` 仍可回退。源代码与远端哈希一致，FoR30/FoR32/FoR36/FoR45/FoR51/validator 窄回归测试全绿。
- FoR51B 重新以当前远端 validator 验证，ID/OOD split 均返回 `formal post-run evidence OK`；主表已从“formal 待补”更新为“formal 已复核、能力仍低于榜首”。

## 79. 增补 68（2026-10-04）：FoR49B 正式复测排队

- 新增独立 `SOTA49B` manifest，ID/OOD 各 4 项、`skip=8`，required tool 为 `z3_check`；远端 preflight 对两个 split 均通过。
- 单卡 4 天任务 `59292461` 已提交，当前 `PENDING/Priority`。结果尚未产生，不能预先计入分数；既有 H49/H49D 工程结果保持原样。

## 80. 增补 69（2026-10-04）：FoR49B 结果与严格验收

- 因账号 8 节点上限，排队的 `59292461` 未使用；同一批 `SOTA49B` 改在已运行 Qwen 单卡服务 `59277346/lrdn2979:37346` 上以 CPU-visible probe 完成，ID/OOD 各 4 个互斥 item，未重复 H49/H49D。
- ID 4/4 通过当前 validator；primary 为 `0.875, 0.875, 0.9375, 0.9375`，均值 **0.90625**。4 个结果都调用 `z3_check`，最终图含 `z3_check -> submit.y`，结果均 `completed/hard_ok/reproducible`。
- OOD 3/4 通过；4 项 primary 为 `0.875, 0.750, 0.875, 0.8125`，全批均值 **0.828125**（通过项均值 0.83333）；第 4 项提交中有 3 个 `unknown`（pure-Z3 diagnostics 标为 4 个 unknown），`z=0`，严格 validator 返回错误并保留为失败，不重跑该 item、不放宽验收。
- 因此 `SOTA49B` 是 **ID formal 4/4、OOD formal 3/4 的部分证据**，不能回写为完整正式 SOTA，也不能用 7/8 的通过项掩盖 OOD 缺口。原有“Z3 等价调用不代表 agent 能力”的可比性结论不变。

## 81. 增补 70（2026-10-04）：FoR49 超时重试路由收紧

- 针对 SOTA49B OOD-11 中“第二个 Z3 节点未接入提交”的可复现轨迹问题，FoR49 objective 已明确困难 QF_NIA 优先使用 `query_timeout_s=120`；如果确需 retry，最新 `z3_check.status` 必须直接成为 `submit.y`，不能保留悬空 solver 节点。
- 修改提交为 `b52d030`；FoR49 定向测试 12 项通过，远端源文件 SHA 与本地一致并完成编译。既有 formal 结果保持不变。

## 82. 增补 71（2026-10-04）：FoR50 冠军权重污染排除

- ValueEval 冠军公开 DeBERTa checkpoint 的模型卡声明训练/测试规模为 9324 条。Leonardo 当前 FoR50 原始文件去表头合计同样为 `5393+1896+1576+279+100+80=9324`，覆盖项目训练、validation、ID、Nahj/OOD、Zhihu 和 NYT 边界。
- 因无法证明 checkpoint 排除了本项目评测标签，该权重不接入 formal/tool-on；只保留 visible-train-only 的 `fit_predict` 和全池诊断。完整来源、计数和决策见 `reports/for50_pretrained_supply_20261004.md`。

## 83. 增补 72（2026-10-04）：FoR38/45/50/52 外部 SOTA 并行审计

- FoR38 staged Chronos-Bolt 完整池 ID/OOD sMAPE 为 `16.45184/10.75699`，固定 macro 为 `13.6593/10.2101`，故不接入。
- FoR45 官方 Gators 方案依赖 Qwen2.5-VL、Gemini RAG 和外部平行语料；Aya Vision 只有 gated base 所需的 PEFT adapter，当前没有可部署的完整本地路线。
- FoR50/52 的供应链排除和无同指标 checkpoint 结论见 `reports/for38_45_50_52_external_sota_20261004.md`；未重复正式 item 或修改验收。

## 84. 增补 73（2026-10-04）：FoR32 MASS 候选验证

- MASS checkpoint 正常加载，CPU-only 可见 train-reference smoke 输出两个 dev crop 的 0/1/2 mask，shape 可逆且值域正确。
- 通过任务 `score_dev` 的受控比较，atlas mean DSC `0.69429`，MASS 替换两例后的 mixed mean DSC `0.59858`；因此它没有在本 crop domain 上提供可用的零样本能力，不进入 ToolSpec 或 formal manifest，FoR32 继续 blocked。
- 启动作业的两个早期失败均为脚本/注册问题，修正后 forward 成功；不把它们计为模型失败。证据见 `reports/for30_for32_external_supply_20261003.md`。

## 85. 增补 74（2026-10-04）：FoR49 SOTA49C OOD-only partial formal evidence

- 容量核查确认 ID 池只有 `00–11` 且全部已用；OOD `12–15` 是互斥未测项。manifest 新增 `SOTA49C`（OOD-only, `skip=12`, required `z3_check`），不重复任何正式 item。
- 复用 Leonardo `59277346/lrdn2979:37346` 后，OOD-12/13/15 严格 validator 均通过，primary=`0.8750/0.9375/0.8125`，均 `z=1`，最终工具 lineage 为 `z3_check.status -> submit.y`；三项均值 `0.8750`。OOD-14 在 120 秒 Z3/代码 retry 期间 step `59277346.149` 被显式取消（exit `0:9`），只有 work trajectory，无 scorer receipt；不计分、不重跑。因此当前是 **SOTA49C OOD 3/4 partial formal**，不能升级成完整批次。
- FoR49 仍按“Z3 工具等价调用不代表独立 agent 推理”解释；既有 SOTA49B 分数和验收未改。细节见 `reports/for49_sota49c_20261004.md`。

## 86. 增补 75（2026-10-04）：FoR51 SevenNet-MF-0 供应复核

- Leonardo 已有公开冻结候选 SevenNet-MF-0（`10,322,158` bytes，SHA-256 `81791329b37d445f46b531578c182c41792d98c7814222c9e5dde276402225fd`）；`SevenNetCalculator("7net-mf-0", modal="PBE"/"R2SCAN")` 只读加载成功。官方训练域是 Materials Project PBE(+U)+r²SCAN 晶体弛豫轨迹，属于结构/力势，不是 phonon-label checkpoint。
- 它与现有 SevenNet/CHGNet 一样有 MP 结构先验重叠，不能作为独立 phonon SOTA；当前主库没有 MF modal/cache 路由，未接入、未评分、未触碰 formal item。MACE/MatterSim/MATGL/FairChem/ORB/JMP 在服务器环境无包且无本地权重，不能安全包装。
- 供应核查曾提交但未启动 `59294443`（`MaxNodePerAccount`），已取消；详细证据见 `reports/for51_external_supply_20261004.md`。

## 2026-10-04 并行低分路线收口

- **FoR30**：官方 PRBonn Panoptic-DeepLab visible-dev 8/8 全背景，`PQ+=24.6095`，低于现有 Mask2Former 组合；没有安全替换，正式 ID/OOD 不重复。详见 `reports/for30_panopticdeeplab_audit_20261004.md`。
- **FoR32**：数据池和评分门禁已复核，没有发现切分或评分器 bug；当前阻塞点是无合规的独立冻结 3-D anterior/posterior 权重。BiomedParse 为 gated 2-D 路线，HSF/Hippodeep 输出定义不匹配，Task04 权重全部排除。详见 `reports/for32_external_candidates_20261004.md`。
- **FoR36**：Sony MIMO-SCNet small visible-dev SDR `8.528417 dB`，较 `htdemucs_ft` `8.091974 dB` 提升 `+0.436443 dB`；已封装为可选 `separate_scnet`，不改既有 formal 分数。详见 `reports/for36_scnet_supply_20261004.md`。
- **FoR45**：Gators/Aya/Florence/SigLIP 均不能作为公平替换（缺完整权重或 API、训练数据与评测集重合、或缺目标语言契约），CLIP 继续 diagnostic-only。详见 `reports/for45_external_route_audit_20261004.md`。
- **FoR50**：数据、切分、VALUES 列和官方 F1 实现均无 bug；全池 visible-train-only F1 为 ID/OOD `0.548950/0.437082`，低分由 16-item 波动和 OOD 分布造成。公开 DeBERTa 训练规模覆盖本项目文件，继续排除。详见 `reports/for50_score_audit_20261004.md`。
- **FoR49**：solver/retry/submit lineage 已修复并通过 12 项定向测试；`SOTA49B` 和 `SOTA49C` 的已有证据仍分别为部分批次，ID 池耗尽且没有可补齐的 paired formal item。详见 `reports/for49_solver_closeout_20261004.md`。

## 2026-10-04 FoR52 domain-adaptive route

FoR52 新增 `psych_domain_adaptive_predict`：只依据 label-free payload 中的 study ID 选择模型——study 出现在 visible `load_train` 时使用 qlearn，未出现时使用 pooled gbdt。完整可见池诊断（144 个 train sessions）为 ID 263 项 qlearn/domain **0.63498**、gbdt **0.55894**；OOD 544 项 qlearn **0.62868**、gbdt/domain **0.65625**。该工具不读取 split flag、target 或 scorer output，旧 fixed formal route、manifest、acceptance 和已有分数均不改；这是可部署的工程优化，不是新 formal SOTA。见 `reports/for52_domain_adaptive_route_20261004.md`。

## 2026-10-04 增补：FoR51 SevenNet-MF-0 路由已部署

FoR51 现在有独立的 SevenNet-MF-0 PBE/r2SCAN 冻结工具路线：模型名和 cache key 分离，`fit_sevennet_mf0_mlip` 明确选择 modal，只拟合 visible train labels。Leonardo 的公开 checkpoint SHA-256 为 `81791329b37d445f46b531578c182c41792d98c7814222c9e5dde276402225fd`，PBE/R2SCAN loader、远端 py_compile 和本地/远端 23 项定向测试通过。尚无 visible-dev 分数，因此不把它写成新成绩；SevenNet-l3i5 formal ID/OOD 结果不变。证据：`reports/for51_sevennet_mf0_route_20261004.md`。

## 2026-10-04 增补：FoR38 TimesFM 2.5 外部路线

官方 TimesFM 2.5 已在 Leonardo 隔离部署并完成 64 ID + 141 OOD 的 label-free trusted-pool 比较；raw sMAPE `15.12694/11.26845`，劣于固定 macro `13.6593/10.2101`，变换输入也未恢复差距。没有发现 scorer、切分或目标遮蔽 bug，因此不注册新工具别名、不重复 formal item。证据：`reports/for38_timesfm25_audit_20261004.md`。

## 2026-10-04 增补：FoR51 MF-0 缓存落盘修复

FoR51 MF-0 预热脚本已修复新缓存目录不自动创建的问题（`e498f57`、`08bdc69`）。Leonardo `59296763` 的 PBE/R2SCAN 4+4 smoke 全部 finite，并把 R2SCAN rows 0–16 实际写入 16/16 缓存；定向测试 26 项通过。该证据证明工具与缓存可部署，不产生新的 formal 分数。

## 2026-10-04 增补：FoR50 SOTA50C 互斥 formal 证据

`SOTA50C` 使用未消耗的 `skip=12` ID/OOD 各 4 项，required tool `fit_predict`。ID primary 均值 `0.445312`、3/4 accepted；OOD primary 均值 `0.239440`、0/4 accepted。8 项均完成、hard-valid、可复现且在预算内，z=0 全部是正常 acceptance 失败。该批次不修改 scorer/验收，也不重复旧 item；它把 FoR50 OOD 的低分和数据分布难度进一步固定为正式证据。

## 2026-10-04 增补：FoR47/30/32 并行核查

FoR47 的 H6 `00–03` 已是旧工程评测，不能重复；互斥 SOTA47B `04–07` 已完成 ID/OOD 8/8 正式 tool-lineage。FoR30 服务器 visible-dev 对比 Weyler `PQ+=27.190359`、Mask2Former `75.386581`，不替换；FoR32 没有新的合规独立 3-D anterior/posterior 权重，继续 blocked。报告分别为 `reports/for47_formal_capacity_20261004.md`、`reports/for30_32_supply_audit_20261004.md`。

## 2026-10-04 增补：FoR52 自适应路线 formal 纠错

`SOTA52C` OOD `16–19` 的四条轨迹误调用了 `psych_fixed_predict`，不能计为
`psych_domain_adaptive_predict` 证据；ID `skip=16` 超出 16 项 ID 池，未产生结果，
现已从 manifest 删除该 split，且不重跑已消耗条目。提交 `de71b8b` 让正式启动器从
manifest 自动向 episode 注入 required tool 约束。

随后用未使用的 OOD `20–23` 完成 `SOTA52D`：四项均有 adaptive tool 到
`submit.y` 的 lineage，primary `0.6250/0.8750/0.6250/0.6875`，均值
**0.703125**，`2/4` 接受；两个 `z=0` 仅为正常验收边际失败。不能把该准确率
宣称为与发表 NLL 同口径的 SOTA。详见 `reports/for52_sota52c_d_20261004.md`。

## 2026-10-04 增补：FoR52 池容量门禁

Leonardo 重新构建 FoR52 split plan 后确认 ID 容量为 **16**、OOD 容量为 **34** 个 episode。
`f29a349` 将容量写入 formal manifest，并在 `check_formal_manifest.py` 中拒绝越界的
`skip+n` 请求；新增回归测试和远端 preflight 均通过。此前 `SOTA52C` 的空 ID 请求因此被
定位为启动门禁缺口，不再被解释成零分或缺失模型能力。

## 2026-10-04 FoR45 USP 路线追加审计

公开的 USP 仓库 `rmaacario/americasnlp2026-usp` 提供 Qwen3-VL-8B → NLLB-200 的两阶段方案，但没有发布训练后的 NLLB checkpoint；其 notebook 仅把模型写入 Kaggle/Colab 临时目录，且视觉阶段不在 Leonardo 工具池。仓库报告基础 NLLB 的 Guaraní dev chrF++ 为 **19.49**、微调后为 **17.57**，同时说明 Bribri/Maya 目标 token 不在基础词表。按“不自行训练、只接入完整冻结权重”的边界，这条路线排除，不启动 formal。

本轮重新对照 AmericasNLP 官方 `baseline/eval.py`：本地 `chrf_pp` 同样使用 `CHRF(word_order=2)`，每条预测先去首尾空白后逐条评分再求均值。FoR45 CLIP 的 ISO fallback、固定 top-3 caption-medoid alias、权重 SHA 和离线加载 guard 均有定向测试；没有发现 scorer、split 或工具到 `submit.y` 的接线问题。完整证据见 `reports/for45_external_route_audit_20261004.md`。

## 2026-10-04 增补：FoR32 外部权重最后检索

本轮检索没有发现可公平替换当前 FoR32 adapter 的冻结公开权重。PAM 的官方 README 明列 D32 MSD Task04 Hippocampus，Nicolik/Mahdyy02 也直接基于 MSD Task04；Hippo-Net 报告的 260 个 T1w 与本地 Task04 260 pool 同源，均排除。HSF/Hippodeep/InnerEye 没有 anterior/posterior 0/1/2 crop 契约，HippUnfold 虽使用独立 HCP-YA 数据，但需要 whole-brain 到 0.3-mm coronal-oblique crop，输出 tissue/subfield/AP coordinate，不能直接替代本任务已裁 1-mm volume。没有下载或包装这些不合规路线，也没有重复 formal item；FoR32 继续 `blocked`。证据矩阵见 `reports/for32_external_weight_search_20261004.md`，提交 `779474c`。

## 2026-10-04 增补：FoR36 SCNet formal 启动器错误

为验证 visible-dev 上较 `htdemucs_ft` 高 `+0.436443 dB` 的 `separate_scnet`，
`SOTA36B` 预留了 ID `04–07`。第一次共享 step 无法启动；第二次在单卡服务
`59264793/22200` 上启动时，服务实际 model id 为 `sc-llm`，启动器错误使用
`sc-llm-l3`，四项全部在 policy step 0 HTTP 404，未产生工具或 scorer 结果。

该批次不计分、不重跑。提交 `64bb53e`、`73f30ac` 已让启动器自动读取 `/v1/models`
并拒绝已有 formal receipt 的重复运行。详细记录见
`reports/for36_sota36b_scnet_formal_20261004.md`。

## 2026-10-04 增补：FoR30 外部 SOTA 权重缺失

HierarchicalMask2Former 论文的 PQ+ **81.89** 以及 CVPPA SAM/HQ-SAM 的高分路线都没有公开可直接加载的 PhenoBench checkpoint；仓库仅含训练代码、配置和数据处理脚本，按“不自行训练”边界不能接入。通用 SAM2/DINOv2 也没有当前 hierarchical 三数组契约。FoR30 继续使用已验证的 Mask2Former，不重复 formal item。详见 `reports/for30_external_sota_supply_20261004.md`。
