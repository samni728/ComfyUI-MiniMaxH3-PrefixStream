# RTX 4060 Ti (8GB VRAM) 生存指南与「H3 拍场面 × LongCat 唱整段」双引擎 MV 工业化架构

本文档基于对 Windows 11 实机节点 **`stone-winpc`（NVIDIA GeForce RTX 4060 Ti 8GB WDDM，8188 MiB 总显存）** 的实测审计，结合 SI 视频实验室《MiniMax H3 150 字四句结构实测》（91% 服从度）、美团 `LongCat-Video-Avatar 1.6` 以及四级质量门禁体系编写。

---

## 一、`stone-winpc` RTX 4060 Ti (8GB) 实机显存审计结论

在 Windows 11 WDDM 桌面环境下实测：

| 指标 / 权重组件 | 实测体积 / 显存表现 | 关键风险与边界 |
| :--- | :--- | :--- |
| **RTX 4060 Ti 物理总显存** | `8188 MiB` | Windows DWM + 浏览器/后台常驻占用 `1.0 ~ 1.5 GiB`（不清空模型时常驻可达 `6518 MiB`） |
| **推荐启动参数** | `--listen 0.0.0.0 --port 8188 --reserve-vram 0.5 --lowvram` | 必须为 Windows WDDM 预留 `0.5 GB`，防止驱动级共享内存雪崩减速 |
| **MiniMax H3 Fused Turbo8 INT8** | `minimax_h3_fused_refdelta_r1024_turbo8_mystic07_int8_convrot.safetensors` (`20.98 GB`) | `608×352`、124 帧（5.17s）、4 步采样实测耗时 **`45.6s`**，峰值显存 **`7343 ~ 7373 MiB`**（距 OOM 仅余 ~800 MiB！） |
| **Qwen3-VL-4B INT8 文本编码器** | `qwen3-vl-4b-heretic_int8_dynamic_convrot.safetensors` (`4.84 GB`) | 绝不可与 20.98GB H3 DiT 同时常驻 8GB 显存！中间必须插入 **`H3VRAMStageGate8GB`** |
| **H3 Video VAE INT8 + Audio VAE** | `3.17 GB` + `605 MB` | 采样结束后必须先过 **`H3VRAMStageGate8GB`** 卸载 DiT，再用 **`VAEDecodeTiled`** 分块解码 |
| **FLUX.3 Klein INT8 + Qwen3.5-9B** | `5.83 GB` + `9.75 GB` | 用于 Gate A 首帧定妆与关键图生成，生成完毕进入视频阶段前必须彻底驱逐 |
| **Seed-VC 歌声转换套件** | `seed-vc` + `bigvgan_v2_44khz` + `roformer` + `whisper-small` | 适合在 Stage 0 一次性导出 `44.1kHz / 48kHz` 完整干声/母带 WAV，随后清空显存 |
| **LongCat-Video / Avatar 1.6 (13.6B)** | 原生 BF16 约 `27 GB`（FP8 约 `14 GB` + Whisper-large-v3） | **8GB 显卡严禁直接裸跑 BF16 原版！** 推荐放到 **Stage-2 Colab NVMe 大显存云端**（A100/L4），或仅用重度量化 + BlockSwap 跑低清口型草稿 |

---

## 二、为什么「141 秒整首歌连续续写 13 段」必崩？如何根治？

在传统长链续写中，误差遵循反馈动力学：
$$\text{Error}_{k+1} \approx A \cdot \text{Error}_{k} + \epsilon_{\text{new}}$$

1. **视觉累积劣化（A12 痛点）**：连续 Prefix 续写超过 4~5 段后，高频锐度衰减或局部皮肤/背景中频斑纹（8~32px）放大，冷暖色温逐段偏移，到第 13 段人物五官与色调严重漂移。
2. **800 字长提示词自毁**：把完整世界观、五张参考图规则与整段歌词动作塞进 1800 字符单镜 Prompt，服从度从 **91% 暴跌至 60%**，且「闭唇倾听」与「开口演唱」指令互杀（A30 痛点）。
3. **逐段拼接音频累积误差**：每段经 Audio VAE 重解码拼接会产生微秒级网格舍入与音质发糊。

### 四重根治方案（本插件 V2 全内置）

1. **「LongCat 唱整段 (A-Roll) + H3 拍场面 (B-Roll)」双引擎分工**：
   - **A-Roll 主唱口型镜头**：交给针对长时口型与防误差累积专门训练的 `LongCat-Video-Avatar 1.6`（30fps）。
   - **B-Roll 剧情/空镜/闭唇反应镜头**：交给光影与运镜极强的 `MiniMax-H3`，采用 **`<= 3 段` PrefixStream 微链（约 12~15 秒）**；每满 3 段由 **`H3PrefixDriftQAProbe`** 触发 `recommend_anchor_reset = True`，切回全新的 FLUX 关键帧锚点！
2. **无痕视频拼接三件套（针对拼接跳切与偏色）**：
   - **`H3PrefixSmartSeamSelector`**：不在第 0 / 39 帧盲目硬切，而是在重叠窗口内自动寻找像素、明度、Sobel 边缘和运动速度最平顺的帧对（`max_phase_shift=0` 时总帧数与音画时长 **100% 不变**）。
   - **`H3PrefixColorMatchStitch`**：在感知均匀的 **CIE L\*a\*b\*** 空间对齐前后镜头的明度与冷暖色度均值/方差，并支持按 `anchor_pull_strength` 向首镜定妆图拉回色准 + **纯 PyTorch 多尺度光流双向微融合 (`optical_flow_warp`, 2~6 帧)** + 零爆音余弦音频淡化。
   - **`H3PrefixDriftQAProbe`**：实时输出 `seam_jump_score`、`sharpness_ratio_vs_anchor`、`color_drift_score` 与 `recommend_anchor_reset` 熔断信号。
3. **`H3FourSentencePromptNode`（150 字四句结构编译器 + 矛盾指令防火墙）**：
   - 严格执行：**第1句 主体+外观锚点（≤30字，设定与指令分离）** → **第2句 具体可拍微动作+场景（自动将“很纠结”改写为“指尖轻敲桌面，目光微垂”；`listening_reaction` 模式强制锁死闭唇倾听）** → **第3句 景别/运镜/光线三选二** → **第4句 风格质感收尾**，负面约束自动截断 ≤2 条，自动拦截「逆光剪影 + 面部清晰」等互杀指令。
4. **`H3LongCatMVMontageStitcher`（单一母带锁死，0.0ms 音画零漂移）**：
   - 将 30fps LongCat A-Roll 与 24fps H3 B-Roll 自动统一帧率、画幅（`cover_crop` 无黑边）与 LAB 色温，**全程直接锁死唯一权威 `48kHz / 44.1kHz` 主歌曲 `AUDIO` 张量**，3 分钟成片音画累积漂移恒为 **`0.0 ms`**！

---

## 三、Stage-1 本地 4060 Ti (8GB) 预演 → Stage-2 Colab NVMe 高清母带双阶段流程

1. **Stage 0 & Gate A（本地 8GB 零成本锁锚点）**：
   - 运行 `Seed-VC` 生成整首歌曲人声母带 WAV → 过 `H3VRAMStageGate8GB` 清空显存。
   - 运行 `FLUX.3 Klein INT8` 生成角色定妆卡与每条微链的首帧关键图，人工肉眼挑刺通过 Gate A → 过 `H3VRAMStageGate8GB` 清空显存。
2. **Stage 1（本地 4060 Ti 8GB 低清动作与剪辑预演）**：
   - 加载 `examples/MiniMaxH3_4060Ti_8GB_3Seg_PrefixStream_MicroChain.json`（`608×352`，4 步 INT8 Turbo，每段 ~46 秒）。
   - 用 `H3PrefixDriftQAProbe` 筛除跳切/糊片，确认动作节奏满意后保存 latent 与时间轴 JSON。
3. **Stage 2（Colab NVMe 云端高清精修 + LongCat 1.6 演唱合成）**：
   - 利用 `aria2c` 16 线程直拉权重至 Colab `/content` 本地 NVMe（抛弃慢速网盘挂载）。
   - 渲染 `LongCat-Video-Avatar 1.6` 720p/30fps 演唱 A-Roll，以及 H3 二阶段高清 B-Roll。
   - 拖入 `examples/H3_LongCat_DualEngine_SoloMV_Master_Template.json`，一键完成双引擎 LAB 调色拉齐与 48kHz 母带零漂移终混，终审通过后立即断开释放云端 GPU。
