# SuperGlue COLMAP 使用说明

脚本：`colmap/superglue_colmap.py`

作用：用 `colmap/SuperGluePretrainedNetwork` 替代 COLMAP 自带 matcher，估计每帧相机的内参、外参、稀疏点云，并把光流估计出的速度写入 PLY。

> 大多数参数已经设了适合本数据集（约 100 路相机、4K 图像）的默认值，**正常情况下只需要几个参数**，见第 3 节。需要调的参数在第 5 节有详细解释。
在 4 卡机上
conda run --no-capture-output -n gsstatic python colmap/superglue_colmap.py --frames 1:201 --static_rig --force `
  --images_root .\data\two\images\ --output_root .\output\twopeople --resize -1 `
  --gpus 0,1,2,3

单帧运行：
conda run --no-capture-output -n gsstatic python colmap/superglue_colmap.py `
    --frames 1 --force `
    --images_root .\data\two\images\ --output_root .\output\two-test `
    --resize -1 `
    --no-compute_velocity --no-compute_flow


(gsstatic) PS D:\LS\guassian_static> python colmap\superglue_colmap.py `
>>   --images_root "Y:\dataset\volygon\teaser\take03\footage\Stills" `
>>   --input_layout cameras `
>>   --output_root Y:\dataset\volygon\teaser\take03\calib `
>>   --frames 1 `
>>   --device cuda `
>>   --resize 2560 `
>>   --no-compute_flow `
>>   --no-compute_velocity `
>>   --force
---

## 1. 数据放法

图像目录（每个子目录是一个时间帧，里面是同一时刻各路相机的图）：

```text
data/twopeople/images/1/1.png      # 帧 1，相机 1
data/twopeople/images/1/2.png      # 帧 1，相机 2
data/twopeople/images/2/1.png      # 帧 2，相机 1
```

也支持“每个子目录是一台相机，目录内是一帧或多帧”的布局。加
`--input_layout cameras` 后，脚本会按自然顺序排列相机目录和各目录内的帧，
自动转置为逐帧输入；相机图像统一命名为 `1.png、2.png、...`，并在输出根目录
写入 `camera_mapping.json` 记录数字文件名与原相机目录的对应关系。所有相机目录
必须具有相同帧数。

```powershell
python colmap/superglue_colmap.py `
  --images_root "Y:\dataset\volygon\teaser\take03\footage\Stills" `
  --input_layout cameras --frames 1 `
  --output_root output/volygon_take03 `
  --no-compute_flow --no-compute_velocity --force
```

> 光流**不再需要你预先准备**——默认（`--compute_flow`）脚本会用 WAFT 自动生成到 `output_root/flows/`（见第 4 节「一条命令跑通」）。只有在 `--no-compute_flow` 的旧模式下才需要外部光流目录 `--flows_root`（`.npy` 形状 `(H,W,2)`）。

`--images_root`、`--output_root` 都有默认值，**只要在仓库根目录运行脚本，就基本不用手动写路径**。

---

## 2. 先跑一个小测试

第一次用先确认环境和流程没问题：只取第 1 帧的前 12 张图、20 对匹配。

```powershell
conda run -n A2PM-new python colmap/superglue_colmap.py --frames 1 --max_images 12 --max_pairs 20 --output_root output/twopeople_superglue_test --force
```

如果 Windows 上 `conda run` 报临时文件占用，直接用环境里的 python：

```powershell
C:\ProgramData\miniconda3\envs\A2PM-new\python.exe colmap/superglue_colmap.py --frames 1 --max_images 12 --max_pairs 20 --output_root output/twopeople_superglue_test --force
```

---

## 3. 正式运行（推荐用法）

相机阵列是**固定不动**的（各帧之间相机位置不变），所以推荐加 `--static_rig`：先用一帧把所有相机的内外参解算出来，再用同一套相机位姿去三角化每一帧。这样：

- **所有帧共享同一套内参 + 外参**（在参考帧上解算并用对极约束清洗后重解一次更新），帧与帧之间天然对齐、同一尺度；
- 每一帧只用已知位姿清洗匹配并三角化点云，**不再改动内外参**（`--rig_fix_poses` 默认开启，位姿全程锁死）；
- 位姿只解一次，更准、更稳，点云噪声更小。

> 参考帧默认用所选的第一帧，可用 `--rig_ref_frame <帧名>` 指定，建议选**人物清晰、遮挡少、相机覆盖好**的一帧——它的位姿质量决定所有帧。

```powershell
# 处理第 1~3 帧（推荐）
conda run -n A2PM-new python colmap/superglue_colmap.py --frames 1:3 --static_rig --force

# 处理全部帧
conda run -n A2PM-new python colmap/superglue_colmap.py --static_rig --force

# 只处理单帧（此时 static_rig 等同于普通重建）
conda run -n A2PM-new python colmap/superglue_colmap.py --frames 1 --force
```

如果相机会移动（不是固定阵列），就**不要**加 `--static_rig`，让每帧各自独立重建：

```powershell
conda run -n A2PM-new python colmap/superglue_colmap.py --frames 1:3 --force
```

> 运行时间提示：**参考帧**用 `exhaustive`（两两全配对）+ 全分辨率解算，最准但最慢（只跑一次）。**后续帧**会自动提速：用参考帧的内外参挑出真正重叠的相机对（共视选对，约 4950→600~900 对），匹配分辨率默认降到 2560（`--rig_resize`），并且每张图的 SuperPoint 特征只算一次复用。整体后续帧通常比参考帧快约 10 倍量级，点云质量基本不变。相关开关见第 5 节。

---

## 4. 输出在哪里

输出已经精简成**直接可喂 3DGS** 的三样东西：

```text
output/twopeople1/
  undistorted/
    images/
      1/                 # 第 1 帧：去畸变 + 主点居中后的图像
        1.png 2.png ...
      2/                 # 第 2 帧 ...
    sparse/
      0/                 # 所有帧共享的一组相机（PINHOLE，主点已居中）+ 外参 + points3D
        cameras.bin images.bin points3D.bin
  points/
    1.ply 2.ply ...      # 每帧稀疏点云；有光流的帧会带速度属性
  velocities/
    1.npz 5.npz ...      # 与 points/<帧>.ply 同名的独立速度文件（只写有光流的帧）
  flows/
    1/                   # 帧 1→2 的前向光流（逐相机）；若 --flow_frame_interval 5 则是 1→5
      1.npy 2.npy ...    # 与 undistorted/images/1/1.png 等逐像素对齐
      _pair.json         # 记录 source_frame / target_frame / flow_frame_interval，防止续跑误复用旧光流
    2/                   # 默认相邻帧时是 2→3；间隔采样时只会有锚点帧目录
```

关键点：
- **`undistorted/sparse/0/` 是所有帧共用的一组内外参**（静态机位），相机模型是 PINHOLE 且**主点已居中**（cx=W/2、cy=H/2），所以原版 3DGS 不会再因为忽略 cx/cy 而发糊。
- **`undistorted/images/<帧>/`** 是对应的去畸变+居中裁剪图像。同一台相机在所有帧里尺寸一致；不同相机尺寸可以不同（都已在 `cameras.bin` 里各自记好）。
- **`flows/<N>/<相机>.npy`** 是 WAFT 在去畸变图上算的前向光流，`(H,W,2)` 原始像素位移，**与 `undistorted/images/<N>/<相机>.png` 逐像素对齐**，直接用作 4DGS 监督。默认是相邻帧 `N→N+1`；如果设置 `--flow_frame_interval 5`，则只记录第 1、5、10... 个选中帧之间的光流，例如 `1→5`、`5→10`。
- **`points/<帧>.ply`** 是每帧点云。只有有光流的源帧会被 velocity pass 更新出 `vx vy vz` 速度属性；非光流锚点帧保持普通点云/零速度。它**仅供你自己的 4D 流程用**；不要拿它当 3DGS 的 input.ply，3DGS 用 `sparse/0/points3D.bin` 初始化。
- **`velocities/<帧>.npz`** 是独立速度文件，文件名与 `points/<帧>.ply` 对应，点顺序也与该 PLY 的 vertex 顺序一一对齐。里面包含 `velocity`、`valid`、`confidence`、`view_counts`、`xyz`、`rgb`、`source_frame`、`target_frame`、`frame_interval`、`velocity_dt`。该文件只会为有光流的源帧写出。

输出帧名与输入帧文件夹名一致：输入 `images/1` → `images/1/`、`points/1.ply`、`velocities/1.npz`（如果第 1 帧有光流）。

喂 3DGS 时，把某一帧组成标准布局即可：`images/` 用 `undistorted/images/<帧>/`，`sparse/0/` 用 `undistorted/sparse/0/`。

> 去畸变+居中是**默认行为**，无需参数。不需要这套导出就加 `--no-undistort`（那样不产生 `undistorted/`）；想压缩图像尺寸可用 `--undistort_max_image_size 2000` 限制长边。

`points/1.ply` 每个点包含：

```text
x y z                  # 坐标
red green blue         # 颜色
vx vy vz               # 速度（COLMAP 世界单位 / (velocity_dt * flow_frame_interval)）
velocity_confidence    # 速度置信度 0~1
velocity_valid         # 该点速度是否有效 0/1
velocity_views         # 参与速度三角化的相机数
```

> `.ply` **默认是 ASCII（文本）格式**，可直接用记事本/编辑器打开查看，CloudCompare、Open3D、3DGS 也都能读。想要更小、读写更快的二进制 PLY 就加 `--no-ply_ascii`（稀疏点云体积差别很小）。

### 一条命令跑通 SuperGlue + COLMAP + WAFT（默认开启）

现在**只输入图片**即可，脚本会自动把三件事串起来，输出 `undistorted/`、`points/`、`flows/`，以及有速度的 `velocities/`：

```powershell
conda run --no-capture-output -n gsstatic python colmap/superglue_colmap.py --frames 1:201 --static_rig --force `
  --images_root .\data\two\images\ --output_root .\output\twopeople --resize -1
```

> ⚠️ **一定要加 `--no-capture-output`**（或先 `conda activate gsstatic` 再直接 `python ...`）。否则 `conda run` 会**缓冲子进程的输出**，进度条和日志都不会实时显示——会让你误以为卡住了。

流程分三阶段（都在一个进程里）：
1. **重建 + 去畸变**：SuperGlue+COLMAP 解算每帧位姿/内参/稀疏点 → 去畸变+主点居中 → `undistorted/images/<帧>/<相机>.png` + 共享 `undistorted/sparse/0`。
2. **WAFT 光流**：重建完成后释放 SuperGlue 显存、加载 WAFT，对**去畸变图**逐相机做前向光流。默认相邻帧 `N→N+1`；设置 `--flow_frame_interval 5` 时只做第 1、5、10... 个选中帧的光流，例如 `1→5`、`5→10`。结果写 `flows/<源帧>/<相机>.npy`。**flow 与 `undistorted/images/<源帧>/<相机>.png` 逐像素对齐**（原生分辨率），可直接做 4DGS 监督。
3. **速度**：在**去畸变 PINHOLE 空间**用刚生成的 flow 给有光流的源帧 3D 点算 `vx/vy/vz`，写进 `points/<源帧>.ply`，并额外写 `velocities/<源帧>.npz`。没有光流的帧不估计初始速度。

> **关于 WAFT 分辨率**：WAFT **不是只能输出 1600×900**——它输出分辨率 = 输入分辨率（内部按 32 的倍数 padding 再裁回）。之前的 1600×900 只是因为喂了 1600×900 的图。这里默认在**去畸变原生分辨率**上跑（与图 1:1）。若显存不够，用 `--flow_max_size 2560` 限制长边（flow 存为该尺寸，按比例对齐）。

关键参数（一般不用动）：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--compute_flow` | 开 | 跑集成的 WAFT 阶段并输出 `flows/`。关掉用 `--no-compute_flow`（退回旧的「外部光流」模式，用 `--flows_root`）。 |
| `--flow_frame_interval` | 1 | 光流锚点间隔。`1` = 保持旧行为，相邻帧 `N→N+1`；`5` = 只记录第 1、5、10... 个选中帧的光流，例如 `1→5`、`5→10`，速度会在结果基础上除以 5。 |
| `--waft_root` | `f:/project/WAFT` | WAFT 仓库路径（进程内导入，需在 `gsstatic` 环境）。 |
| `--waft_ckpt` | `ckpts/a2/waftv2-ckpts/twins/zero-shot.pth` | WAFT 权重（相对 `--waft_root`）。`--waft_cfg` 同理。 |
| `--flow_max_size` | 0（原生） | 限制喂给 WAFT 的长边以省显存；0 = 去畸变原生分辨率。 |
| `--flow_direction` | forward | 速度按前向 `N→N+1` 解释。若另有反向流可用 `backward` 取反。 |
| `--velocity_dt` | 1.0 | 单帧时间间隔。最终速度分母是 `velocity_dt * flow_frame_interval`。通常保持 1.0，用 `--flow_frame_interval` 控制光流时间基线即可。 |
| `--skip_existing_flow` | 关 | 断点续跑时跳过已存在的 `.npy`。脚本会检查 `flows/<源帧>/_pair.json`，只有源帧、目标帧、间隔都一致才复用，避免换间隔后误用旧光流。 |

> 跑完看日志自检：`flow: a->b wrote N/M cameras`、`velocity(undist): N/M points passed`、`velocity(undist): median applied flow displacement = X px`（应是亚像素~几像素，不该是上千）。位移 >10% 图宽会告警。
>
> **提升速度信号**：若逐帧运动是亚像素（信号弱），用更大时间基线，例如 `--flow_frame_interval 5` 会计算第 1→5、5→10... 帧的光流，并自动把速度除以 5；或保持原生分辨率（默认）以保留细节。

---

## 5. 需要了解的参数

下面是**实际可能要改**的参数，其它的保持默认即可。

### 选帧 / 路径 / 覆盖

| 参数 | 说明 |
| --- | --- |
| `--frames 1` | 只处理第 1 帧。 |
| `--frames 1:3` | 处理第 1 到第 3 帧（含两端）。也支持逗号，如 `--frames 1,3,5`。不写则处理全部帧。 |
| `--output_root <目录>` | 输出目录，默认 `output/twopeople_superglue`。 |
| `--force` | 覆盖该帧已有的旧结果。重复跑同一目录时一般都要加。 |

### 静态相机阵列

| 参数 | 说明 |
| --- | --- |
| `--static_rig` | **核心开关**。相机固定不动时加上：用一帧解出的相机位姿三角化所有帧，保证跨帧对齐、降低噪声。相机会动则不要加。 |
| `--rig_ref_frame <帧名>` | `--static_rig` 时用哪一帧解算参考位姿，默认用所选的第一帧。建议选**人物清晰、遮挡少、相机覆盖好**的一帧，参考帧的位姿质量决定了所有帧的质量。 |

### 精度 / 显存（按机器情况调）

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--resize 2560` | 2560 | SuperGlue 处理时把图像长边缩放到的像素数。**越大特征点定位越准、点云越细，但越吃显存、越慢。** 显存不够就调小（如 `--resize 2048`）；想要极致精度且显存足够可用 `--resize 3200` 或 `--resize -1`（原图）。 |
| `--camera_model OPENCV` | OPENCV | 相机模型。默认 OPENCV 会估计镜头畸变，能明显减少边缘噪声。若某些相机点太少导致畸变估计不稳，改成 `--camera_model SIMPLE_RADIAL`（更简单更稳）。 |

### 点云清洗（结果太脏或太空时调）

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--ply_min_track_len 3` | 3 | 只保留被**至少这么多相机**看到的点。值越大越干净但点越少；设 `0` 关闭该过滤。 |
| `--ply_max_reproj_error 2.0` | 2.0 | 丢弃重投影误差大于该值（像素）的点。值越小越干净；设 `0` 关闭该过滤。点太少时可放宽到 `3.0~4.0`。 |

### 提速（--static_rig 后续帧，默认已开启）

参考帧之后的每一帧都会用已解出的内外参来加速，质量基本不变。一般不用动，跑得太慢/太空时再调：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--rig_pair_mode` | covisibility | 后续帧只匹配**真正重叠**的相机对（共视 + 视角相邻），约 4950→600~900 对。想退回每帧全配对用 `--rig_pair_mode same`。 |
| `--rig_resize 2560` | 2560 | 后续帧的匹配分辨率（位姿已固定，降一点更快）。想后续帧也用全分辨率：`--rig_resize -1`（更慢更细）。 |
| `--rig_covis_top_k 12` | 12 | 每台相机保留多少个共视最强的邻居。点云偏空就调大（如 18~24，更稳但更慢）。 |
| `--rig_geo_neighbors 6` | 6 | 每台相机额外按视角相邻补几个邻居，保证物理相邻相机一定匹配（应对人物走动）。 |

> SuperPoint 特征缓存（每张图只检测一次、跨相机对复用）是自动的，参考帧和后续帧都生效，无需开关。

### 相似特征 / 对称场景（自动清理 + 用户辅助）

环形机位下**对称位置的相机看到的内容非常相似**（重复纹理、对称布景、绿幕格子等），SuperGlue 会在这些「长得像但其实不重叠」的相机对之间产生**自信但错误**的匹配，单对的 RANSAC 挡不住，最终 mapper 解出错误位姿或点云重影。脚本现在有一层自动防线 + 四种简单的用户辅助手段：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--cycle_filter` | 开 | **自动防线**：任意三台相机的相对旋转绕一圈应回到原位（R_ca·R_bc·R_ab≈I）。错误对会污染它所在的**每一个**三角形，而正确对至少有一个干净三角形，因此按「最好三角形的环路误差」逐个剔除坏对（每删一个重新评分）。 |
| `--cycle_max_rot_error 8.0` | 8.0 | 剔除阈值（度）。误删好对就调大（12~15），漏删坏对就调小（5~6）。 |
| `--pair_report` | 开 | 每帧输出 `output_root/pair_report/<帧>/`：`pairs_diagnostics.txt`（全部对的匹配数/内点率/环路误差排名）、`suspect_pairs.txt`（被剔除对，**可直接粘进黑名单**）、以及被剔除/可疑对的**左右拼图连线预览 jpg**——肉眼一翻就知道删得对不对。 |
| `--layout_file <文件>` | 无 | **辅助 1（最推荐）**：告诉脚本相机的物理顺序。文本文件按物理环序一行一个相机名（或逗号分隔，`#` 注释，可只写数字不带扩展名）。给了以后只匹配布局上相距 `--layout_window`（默认 10）以内的相机——对面的相似相机**根本不会被配对**，参考帧配对数也从 4950 降到约 1000（更快）。`--layout_ring`（默认开）表示首尾相邻的闭环。 |
| `--pair_blacklist <文件>` | 无 | **辅助 2**：明确禁止某些相机对。一行两个名字（如 `3 57`）。确认 `pair_report/suspect_pairs.txt` 里的对是错的之后直接粘进来，之后每次运行都生效。 |
| `--mask_root <目录>` | 无 | **辅助 3**：每台相机一张掩膜图 `<相机名>.png`（COLMAP 惯例：**黑色=0 的区域不取特征点**）。把重复纹理背景/对称布景涂黑一次即可，静态机位所有帧复用同一套掩膜。 |
| `--init_pair 名0 名1` | 无 | **辅助 4**：指定 mapper 的初始化图像对。当自动初始化选到错误对（`no initial pair` 或初始化就歪了）时，手动挑一对纹理丰富、明显重叠的相机。 |
| `--min_inlier_ratio 0` | 0=关 | 附加门槛：整对的 RANSAC 内点率低于该值就丢弃全对。歧义严重的机位可试 0.2~0.35。 |

**推荐流程**（遇到「特征接近的图组解不对」时）：

```powershell
# 第 1 步：先正常跑一次参考帧，看自动过滤和报告
conda run --no-capture-output -n gsstatic python colmap/superglue_colmap.py --frames 1 --static_rig --force

# 第 2 步：打开 output_root/pair_report/1/ 翻预览图 —— 红线=被剔除的可疑对，绿线=保留但误差偏高的对
#          确认删对了 -> 把 suspect_pairs.txt 内容粘到 blacklist.txt（一劳永逸）

# 第 3 步（强烈推荐）：按物理顺序写一个 layout.txt（每行一个相机名），从源头避免对面相机互配
conda run --no-capture-output -n gsstatic python colmap/superglue_colmap.py --frames 1:201 --static_rig --force `
  --layout_file layout.txt --pair_blacklist blacklist.txt

# 仍不行：涂掩膜排除重复背景（--mask_root masks/），或手动指定初始化对（--init_pair 12 13）
```

> 参考帧解对了，后续帧走共视选对 + 已知位姿对极清洗，天然不受相似特征影响；所以以上手段**主要针对参考帧（和非 static_rig 的独立重建帧）**。

### 压榨硬件（GPU/CPU 重叠，默认已开启）

下面这些只影响**速度和数值精度**，不改变算法/选对策略，目的是把 GPU 和 CPU 同时喂满：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--pipeline` | 开 | `--static_rig` 下，把每帧的 CPU/磁盘 solve（三角化+去畸变+裁剪）和后续帧的 GPU 匹配**重叠**起来。各帧相互独立，输出完全一致。想串行排错用 `--no-pipeline`。 |
| `--solver_workers` | 0=自动 | 流水线里同时跑几个帧的 solve（去畸变+裁剪是磁盘密集型，所以默认按核数取 2 个，避免磁盘抖动）。GPU 会一直往前匹配、不再傻等单个 solve。点云/图像不变。 |
| `--export_workers` | 0=自动 | 每帧去畸变图裁剪/写盘的线程数（cv2，释放 GIL→真多核）。默认按核数取（≤8）。这是之前 GPU 空闲、CPU 也不高的那段「串行裁剪 100×4K」的提速点。 |
| `--decode_workers 4` | 4 | 后台预解码图片的线程数，让磁盘/CPU 的 4K 解码和 GPU 检测重叠。输出不变。机器核多可调大；设 `1` 关闭预取。 |
| `--filter_workers 2` | 2 | 每对匹配后的 CPU RANSAC 清洗放到后台线程，与 GPU 匹配下一对**重叠**（cv2 释放 GIL）。输出不变；设 `1` 关闭。 |
| `--fp16` | 开 | SuperPoint+SuperGlue 用 fp16 混合精度（Ampere/Ada 上约 1.5~2×）。**关键点坐标仍保持 fp32**，匹配点集只有极微差异（后面有 RANSAC + 对极清洗兜底）。要逐位一致的全精度匹配用 `--no-fp16`。 |
| `--tf32` | 开 | 允许 Ampere+ 上的 TF32 矩阵/卷积（小幅加速，数值差异比 fp16 更小）。要绝对精确用 `--no-tf32`。 |
| `--gpus` | 关（单卡） | **多卡数据并行**。传 `0,1,2,3`（或 `all`）后，把「逐帧 SuperGlue 匹配」和「WAFT 逐相机光流」两个 GPU 大头分摊到多张卡上——每张卡一个 worker 进程，帧/光流图相互独立，输出与单卡一致。不传 = 保持单卡（`--device`）行为。参考帧仍单卡解一次。仅 `--device cuda` 生效。 |
| `--colmap_threads` | 0=自动 | 每个 COLMAP mapper/point_triangulator 调用的线程数（`--Mapper.num_threads`）。`0`=自动：单卡不限（沿用旧行为），多卡时取 `CPU核数//卡数`，避免 4 个并发 COLMAP 抢满 CPU。 |

### 多GPU（把整个项目搬到多卡机器上跑）

在有多张卡的机器上，直接加 `--gpus 0,1,2,3` 即可让匹配和光流两个阶段接近 **N×** 加速（N=卡数）：

```powershell
conda run --no-capture-output -n gsstatic python colmap/superglue_colmap.py --frames 1:201 --static_rig --force `
  --images_root .\data\two\images\ --output_root .\output\twopeople --resize -1 `
  --gpus 0,1,2,3
```

- 原理：每张卡起一个独立 worker 进程（`CUDA_VISIBLE_DEVICES` 绑卡），从共享队列里领帧/领光流图，谁空谁领——天然负载均衡。各 worker 只写互不相交的文件（`points/<帧>.ply`、`undistorted/images/<帧>/`、`velocities/<帧>.npz`、`flows/<源帧>/<相机>.npy`），**输出与单卡完全一致**（fp16 的非逐位一致是既有行为，与多卡无关）。
- 参考帧位姿解算跑一次、仍用单卡（`--gpus` 里的第一张卡）；之后的逐帧匹配+三角化、以及 WAFT 光流才多卡并行。速度阶段（CPU）在主进程单次完成。
- 单帧/单光流图若失败，只记 warning 继续，不再整批中断。
- CPU 别抢崩：多卡会并发多个 COLMAP solve，脚本默认按 `CPU核数//卡数` 限制每个 solve 的线程；需要手动可用 `--colmap_threads`。
- **Dense/MVS（`--dense`）另说**：COLMAP `patch_match_stereo` 自己就支持多卡，直接 `--gpu_index 0,1,2,3` 即可（与上面的 `--gpus` 相互独立）。

> 说明：`--fp16` / `--tf32` 是「近似无损」——实测对 SfM 质量基本无影响，但不是逐位相同。若你要求严格可复现，用 `--no-fp16 --no-tf32`，仍可享受 `--pipeline` + 预解码（这两者**逐位一致**）带来的提速。COLMAP 三角化/去畸变本身已默认吃满所有 CPU 核。

### 其它默认已开启（一般不用动）

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--epipolar_filter` | 开 | 用估出来的位姿做对极约束清洗匹配：参考帧会清洗后**重解一次**（更新内外参），其余每帧三角化前也清洗一遍。明显降噪。极少数情况想关用 `--no-epipolar_filter`。 |
| `--epipolar_max_error 1.5` | 1.5 | 对极清洗的像素阈值。越小越严格、越干净；匹配被删太多就放宽到 `2.0~3.0`。 |
| `--rig_fix_poses` | 开 | `--static_rig` 下把每帧位姿锁死为参考帧的位姿（保证全程共享一套外参）。需要 COLMAP ≥ 3.7；若报错会自动退回不锁。 |
| `--undistort` | 开 | 输出去畸变 + **主点居中**的图像和 PINHOLE 内参（3DGS-ready），见第 4 节。不要就 `--no-undistort`。 |

### 其它常用开关

| 参数 | 说明 |
| --- | --- |
| `--no-compute_velocity` | 不估计速度，只输出点云（更快）。 |
| `--progress` / `--no-progress` | 进度条（默认开）：显示帧进度+预计剩余时间、单帧匹配对进度、光流进度。重定向到文件时可用 `--no-progress` 回到纯日志。 |
| `--keep_workspace` | 保留临时图片、database、日志和 TXT 模型，排错时用。 |
| `--single_camera` | 所有相机共用一套内参。本数据集是 100 路**不同**相机，**不要**加。 |

---

## 6. 结果不理想时怎么办

**特征相近/对称机位解不对**（相机位置解错、点云重影或「对穿」、mapper 把对面相机当邻居）：见第 5 节「**相似特征 / 对称场景**」。要点：先看 `output_root/pair_report/<参考帧>/` 的预览图确认哪些相机对被自动剔除；确认后粘进 `--pair_blacklist`；最好再写一个 `--layout_file`（按物理顺序一行一个相机名）从源头限制只匹配相邻相机；还不行就 `--mask_root` 涂掉重复背景、`--init_pair` 手动指定初始化对。

**点云太空 / COLMAP 报 `no initial pair` / 匹配太少**：放宽匹配和建图门槛。若是初始化选错对导致，直接用 `--init_pair <相机A> <相机B>` 指定一对纹理丰富、明显重叠的相机。

```powershell
conda run -n A2PM-new python colmap/superglue_colmap.py --frames 1 --static_rig --min_matches 15 --mapper_min_num_matches 10 --mapper_init_min_num_inliers 15 --force
```

**COLMAP mapper 报 `ba_config.NumImages() >= 2` / `At least two images must be registered for global bundle-adjustment`**：这是 COLMAP 多模型重建时偶发的全局 BA 断言。脚本会自动改用单模型重试；如果你想手动避开，也可以直接加：

```powershell
conda run -n A2PM-new python colmap/superglue_colmap.py --frames 1:201 --static_rig --no-mapper_multiple_models --force
```

**点云还是有噪声**：收紧清洗门槛（更干净）。

```powershell
conda run -n A2PM-new python colmap/superglue_colmap.py --frames 1:3 --static_rig --ply_min_track_len 4 --ply_max_reproj_error 1.5 --force
```

**后续帧还是太慢**：把后续帧分辨率再降一点、共视邻居数再少一点。

```powershell
conda run -n A2PM-new python colmap/superglue_colmap.py --frames 1:3 --static_rig --rig_resize 2048 --rig_covis_top_k 10 --force
```

**后续帧点云比参考帧偏空**（共视选对漏了相机对）：调大邻居数，或退回每帧全配对。

```powershell
conda run -n A2PM-new python colmap/superglue_colmap.py --frames 1:3 --static_rig --rig_covis_top_k 20 --rig_geo_neighbors 10 --force
# 或彻底退回（最稳最慢）：
conda run -n A2PM-new python colmap/superglue_colmap.py --frames 1:3 --static_rig --rig_pair_mode same --force
```

**速度有效点太少**：先看日志的 `velocity: median applied flow displacement` 和 `velocity drops`。若位移是上千像素并有告警 → 是光流格式/空间用错（见上面「WAFT 光流接入」，WAFT 用 `--flow_format pixel`、在**原图**上跑）。确认无误后再放宽门槛：

```powershell
conda run -n A2PM-new python colmap/superglue_colmap.py --frames 1 --static_rig --velocity_min_views 2 --velocity_max_reproj_error 6.0 --force
```

**速度方向反了**（位移合理但 vx/vy/vz 符号不对）：你的光流是反向（N+1→N）。加 `--flow_direction backward`：

```powershell
conda run -n A2PM-new python colmap/superglue_colmap.py --frames 1 --static_rig --flow_direction backward --force
```

**想看中间文件排错**：

```powershell
conda run -n A2PM-new python colmap/superglue_colmap.py --frames 1 --keep_workspace --force
```

---

## 7. 速度是怎么估计的

不是单相机直接投影，而是多视角光流三角化：

1. 读取 COLMAP 中每个 3D 点被哪些相机真实看到；
2. 在这些相机的光流图里采样它从源帧到目标帧的像素位移；
3. 用多个相机的“目标帧像素”射线重新三角化出目标时刻的 3D 点；
4. 用重投影误差逐步剔除错误光流和遮挡视角；
5. 速度 = （目标帧位置 − 源帧位置）/ (`velocity_dt * flow_frame_interval`)，写入 `vx/vy/vz`。例如 `--flow_frame_interval 5` 时，先用 `1→5` 的位移估计速度，再除以 5，得到平均到单帧尺度的初始速度。

这样比单视角投影更稳，可以减少镂空点或错误表面投影造成的伪速度。
