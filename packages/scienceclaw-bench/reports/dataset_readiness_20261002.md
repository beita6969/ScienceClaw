# ScienceClaw 数据集服务器就绪性核验（2026-10-02）

## 验收范围

目标是让 Leonardo 上 FoR30–FoR52 的 23 个数据入口都能被当前适配器加载，并保留真实的数据限制、来源和分割口径。代码/环境放在 `$F=/leonardo_scratch/fast/AIFAC_F02_774/rqian000`，数据、权重和运行输出放在 `$L=/leonardo_scratch/large/userexternal/rqian000`。

## 现场证据

- Leonardo SSH：`login01.leonardo.local`，用户 `rqian000`。
- 全 23 项统一 `check_avail.py`：退出码 0，耗时 146 s。
- 每一项的容量均为 `src=2, val=2, id=4`。
- OOD 为 4 的学科：FoR30–35、FoR37–39、FoR41–43、FoR45–52。
- OOD 按设计为空的学科：FoR36（没有独立 OOD 集）、FoR40（reconstructed_v3 没有未见机器/跨数据集池）、FoR44（交付角色不是跨数据集 OOD）。这三项不是传输或加载故障。

## FoR49 补齐记录

Leonardo 原先唯一缺少 FoR49。Mac 正本是 SMT-LIB 2025 非线性整数算术公开源库的重建版本；它明确不是原始 ScienceClaw 64/64 或 SMT-COMP 2025 精确选集，因此只作为可运行源库，并保留这一限制。

- VPS 分块在拉取前通过 `sha256sum -c` 和拼接 `gzip -t` 校验通过。
- 远端 candidate frame SHA-256：`88ac7fd63ebbf165fc6e32a24f026647145dbc477d67ae23e7033ba5acba8dfc`。
- 远端适配器读取：`6683` 个 lineage units，`19930` 个 eligible 条目，代表文件缺失数为 `0`；`source_library` 当前包含 `19379` 个文件。
- FoR49 单项检查：`src=2, val=2, id=4, ood=4`；清理临时 incoming 和 VPS 分块后再次检查仍通过。
- 运行包只包含适配器所需的 representative 文件、candidate frame 和审计元数据；agent 仍通过现有 adapter 只看到去标签后的 opaque 输入，candidate frame/receipt 不进入 agent sandbox。

## 工具与环境

当前远端关键工具的 smoke/available 检查均已通过：FoR30 的 PhenoBench Mask2Former、FoR36 的 Demucs 和静态 ffmpeg、FoR40 的 AST/CLAP、FoR47 的 Stanza/CamemBERT、FoR48 的文本编码器，以及现有 TSFM、蛋白、医学分割和 Matbench 入口。P3 的 BuildingsBench Transformer-L、BEATs、TabPFN/tabpfn_2_5 和 granite-tsfm 在服务器上仍没有实现或权重，不能把现有 AST/CLAP 等工具冒充成这些组件。
- 22:22 UTC 的登录节点复核进一步确认：模型目录中只有已有 Chronos TSFM 权重；sc-harness 可导入 `transformers`/`torchaudio`，但 `tabpfn`、`beats`、`granite_tsfm` 和 `pytorch_forecasting` 均不可导入。P3 因而仍是“缺依赖与可核验权重”的待办，不是尚未接通的现成工具。
- 22:51 UTC 的补充核验确认：`tabpfn==2.2.1` 的依赖约束与现有 sc-harness 的 `pandas=3.0.6`、`scikit-learn=1.9.1`、`huggingface-hub=1.33.0` 不一致；`tabpfn==9.1.0` 的代码包虽标为 Apache-2.0，但 TabPFN-2.5/2.6/3 系列权重另受非商业许可，首次使用还需 Prior Labs 授权 token。当前没有已授权的本地权重或 token，因此没有把它装进共享 harness，也没有把相近 AST/CLAP/Chronos 组件冒充 P3 工具。下一步应使用独立环境，并在权重许可与校验文件到位后才安装/封装。

Node0015 只保留为旧资产来源和校验位置，不作为新的数据落点；共享文件系统已使用约 91%，本轮没有向其中写入或清理他人目录。

## 仍需保留的限制

FoR35 的 OOD 仍是同库 cohort 口径；FoR45 官方 test captions 不公开；FoR43 官方 masked-test gold 不公开；FoR38 的预训练 vintage 风险不能靠搬运数据消除。它们均已在各自任务文档中记录，不能用伪造标签、随机 OOD 或未授权数据下载来“补齐”。

## 2026-10-03 P3 资产分阶段落地

在 Leonardo 的 `$L=/leonardo_scratch/large/userexternal/rqian000/p3-stage-20261003` 建立了独立资产目录，没有写入共享 `sc-harness`，也没有训练或读取评测标签：

- BuildingsBench `Transformer_Gaussian_L.pt`：`1,930,570,777` bytes，SHA-256 `fdb4ecf45568d0c2467cd67dd00e53c29e48a4b4ae7d39c814af4dc4f907e30f`，来源为 OEDI S3，许可证按 BSD-3/BuildingsBench 条款记录。
- BEATs `BEATs_iter3.pt`：`361,499,833` bytes，SHA-256 `8d1b234032a9ccff353612dc6c20982346dc2968b205b79d97303eb5e77bfb34`，HF revision `5b53b0404df452a3a607d7e67687227730e5bad1`，MIT；交接中失效的 GitHub 地址没有再使用。
- Granite TTM r2：`model.safetensors` `3,240,592` bytes，SHA-256 `a706726a7eb01bbcb42994b7dcb3c06ea9557898dbae8d480eb04fe8ccb89710`，HF repo `ibm-granite/granite-timeseries-ttm-r2`，Apache-2.0。其模型卡没有明确承诺 6 小时频率，因此只做组件 smoke；FoR37 接入前必须先在 src/val 预先确定重采样策略。
- TabPFN 未下载：权重的非商业许可与 Prior Labs 授权 token 仍缺，不能在无授权条件下安装或封装。

BuildingsBench 和 BEATs 已完成独立环境 GPU smoke，但仍是 `smoke_validated_unintegrated`；Granite 已增加只读 smoke wrapper 并完成独立环境前向，三份资产均不计作任何正式分数。只有在 wrapper 和 src/val 选择完成后，才可进入新的、不重复的 formal manifest。

`59278302`（BEATs）在 normal 单卡上用 pinned Microsoft/unilm `31c5b904…` 完成随机 1 s/16 kHz waveform 前向，输出 `(1,48,768)`；`59278303`（BuildingsBench）用 `v1.1.0` `c60e25f…` 的官方 Gaussian-L 类完成随机前向，输出 `(1,24,2)`。两项均退出码 0，没有读取 FoR33/FoR40 target，也没有写入 scorer。

## 2026-10-03 P3 独立环境 smoke 复核

- Granite 独立环境 `$F/envs/p3-granite-20261003` 已安装 `granite-tsfm==0.3.10`；包的实际 Python 模块名为 `tsfm_public`，不是 `granite_tsfm`。
- 使用本地已校验的 Granite r2 目录和 `local_files_only=True` 成功加载 `TinyTimeMixerForPrediction`，配置为 context `512`、prediction `96`；随机输入完成一次前向，输出形状为 `(1, 96, 1)`。
- 这只证明权重、依赖和基础推理链路可用，不代表 FoR37 的 6 小时频率已经适配，也不产生 formal 分数。FoR37 仍需先在 src/val 固定重采样/频率策略，再封装工具并建立独立 manifest。
- `scilib/granite.py` 已同步 Leonardo，默认只接受本地 checkpoint 目录，始终使用 `local_files_only=True`；独立环境真实 smoke 为 `available=True`、前向 `(1,512,1)->(1,96,1)`、有限输出。该模块是供应链验证接口，不是 FoR37 scorer 或 formal tool。

### FoR37 Granite 频率复核（2026-10-03）

在 Leonardo 仅调用 FoR37 的 `load_train` 和 `load_dev`：训练输出是 1336 个 6 小时步长的 `(64,32)` 场，dev 输出是 16 个 `(4,64,32)` 上下文。Granite r2 的本地配置是 512→96、单通道，模型卡只支持 minutely/hourly，并明确不建议 upsampling/zero-padding。原生 6 小时输入对应 128 天 context 和 24 天 forecast，不符合 24 小时（4 步）任务；把四个 visible dev 场插值到 hourly 也没有连续 512 小时输入，且会改变协议。因此 Granite 继续标记为 `smoke_validated_unintegrated`，不进入 FoR37 formal。可复现证据在 `reports/for37_granite_frequency_20261003.md`。
