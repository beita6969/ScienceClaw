# P3 单卡长作业执行建议（2026-10-03）

本记录只覆盖只读供应链/smoke，不计 formal 分数，不读取 target，不训练，也不重复已评 item。

## 当前已核验

- Leonardo `$L/p3-stage-20261003` 中 BuildingsBench `Transformer_Gaussian_L.pt` 为 1,930,570,777 bytes，SHA-256 `fdb4ecf45568d0c2467cd67dd00e53c29e48a4b4ae7d39c814af4dc4f907e30f`；文件可作为 PyTorch zip 归档读取，包含 1,519 个条目、约 1,930,230,025 个归档字节。完整 `torch.load` 需要大内存和较长启动时间，登录节点上的 metadata-only 尝试没有在 30 s 内完成，因此不把它误报成已完成前向 smoke。
- BEATs `BEATs_iter3.pt` 为 361,499,833 bytes，SHA-256 `8d1b234032a9ccff353612dc6c20982346dc2968b205b79d97303eb5e77bfb34`。在共享 `sc-harness` 中只读载入成功，顶层为 `cfg/model`，12 层、768 hidden、约 90,354,032 个参数；该环境没有 `BEATs`、`beats` 或 `fairseq` Python 模块，因而尚无真实 waveform forward wrapper。
- TabPFN 仍没有授权 token 或可用的权重许可，继续不下载、不安装、不申请 formal 工具。

## 推荐的并行单卡作业

1. **BuildingsBench（1×A100，至多 1 天，`bprod`）**：在独立环境安装与 checkpoint 相容的 BuildingsBench commit，先在计算节点做 `torch.load(map_location='cuda')`、参数键/形状统计和一个随机 `(batch, 168, features)` 前向；禁止导入 FoR33 targets，输出只写 checkpoint SHA、依赖版本、输入输出 shape。若模型接口仍要求官方特征工程，保存“缺少可复现官方 preprocessing”的结果，不接入 scorer。
2. **BEATs（1×A100，至多 1 天，`bprod`）**：固定 HF revision `5b53b0404df452a3a607d7e67687227730e5bad1`，在独立环境补齐上游 BEATs 推理代码（仅代码，不下载 DCASE 标签或微调权重），执行 1 s/16 kHz 随机 waveform 的 CPU/GPU 前向，记录中间层 shape 和显存峰值。只有输入采样率、层池化和输出 shape 能在 src/val 预先确定，才另行写 adapter；此 smoke 不产生 FoR40 分数。
3. **TabPFN**：不提交作业。先由用户确认 Prior Labs 非商业授权和 token；授权到位后再用 1×A100、1 天独立环境做 import/合成表格前向，不能在授权前读取或缓存权重。

当前 `boost_qos_lprod` 已有两张 4 天 Qwen 单卡服务且占用 6/8 节点；P3 smoke 不需要长 Qwen 卡，优先申请空闲 `bprod` 单卡。若只能用长卡，可将上述两个只读 smoke 合并为两个互不共享缓存的单卡作业，每个作业最长 24 小时即可，无需申请 2–4 天。

## 交付边界

上述 smoke 只验证“权重、依赖、推理链路可用”。BuildingsBench 与 BEATs 标记为 `smoke_validated_unintegrated`；没有得到可复现的官方 preprocessing 和 src/val 选择前，不改 formal 状态，不启动新的 FoR33/FoR40 item。

## 2026-10-03 实际执行结果

- `59278302`（normal，1×A100）完成 BEATs smoke，使用固定 Microsoft/unilm `31c5b904ca1bf2afb4c234a6675c683a4e5fc7cd` 的 `BEATs.py` 和已校验 `BEATs_iter3.pt`；随机 1 s/16 kHz waveform 在 GPU 前向输出 `(1,48,768)`，退出码 0。
- `59278303`（normal，1×A100）完成 BuildingsBench smoke，使用 `v1.1.0` `c60e25f37dbae8f3664df0879aaa13a7fe25952d` 的 `LoadForecastingTransformer`、官方 Gaussian-L 配置和已校验 checkpoint；随机输入前向输出 `(1,24,2)`，退出码 0。
- 两个作业只读取冻结权重和官方代码，未读取 FoR33/FoR40 数据或 target，也没有训练、scorer 或 formal episode。资产状态提升为 `smoke_validated_unintegrated`；仍须补齐官方 preprocessing 与 src/val 选择后，才有资格做新的不重复 formal 工具接入。

## 2026-10-03 主库 wrapper 复核

- 作业 `59279440` 在 normal 单卡上直接调用 `scilib.beats.encode` 与 `scilib.buildingsbench.forecast`，而不是调用旁路脚本；两者分别输出 `(1,48,768)` 与 `(1,24,2)`，`finite=true`，退出码 0。
- 该 smoke 仍只使用随机可见输入和显式本地 checkpoint/source，未读取任何 FoR33/FoR40 target。两个 wrapper 已有离线契约测试，但不自动注册为正式 evaluator 或接受线工具。

## 2026-10-03 作业提交复核

已同步 `scripts/leonardo/p3_asset_smoke.py` 与 `p3_asset_smoke.sbatch` 到 Leonardo。两次实际提交（`P3_MODE=buildings`、`P3_MODE=beats`）均被 Slurm 拒绝，返回 `QOSMinCpuNotSatisfied` / `job submit limit`；将 CPU 调整为 8、16、32，显存调整为 100–400G，仍被拒绝。`boost_qos_bprod` 当前账户关联存在，但其 QoS 的 `MinTRES` 是集群级批量资源约束（CPU/GPU/node），不允许这类单卡请求。没有生成 job ID，也没有伪造“已排队”状态。

一次 `--test-only` 的 lprod 查询显示单卡 1 天最早调度到 2026-10-09，未实际提交。待管理员/用户释放 bprod 单卡配额或指定允许单卡的 partition/QOS 后，可直接重提上述两个模式；脚本和 checkpoint 路径已固定。
