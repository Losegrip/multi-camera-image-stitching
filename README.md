# multi-camera-image-stitching

Multi-camera image stitching with cylindrical projection, seam-aware blending
and grid-based illumination compensation.  
The pipeline is modular and **video-ready**: all components can be reused
for real-time multi-camera video stitching.

> 简介：多路摄像机全景图像拼接工程，包含柱面投影、接缝线优化、
> 简单羽化融合以及网格光照补偿模块，代码结构为后续升级到实时视频拼接预留了接口。

---

## Features

- **Multi-camera image stitching**
  - 支持多路图像输入与统一“世界画布（panorama canvas）”计算
- **Cylindrical projection (optional)**
  - 可选柱面投影（CP-SIFT 风格），适合较大视场角场景
- **Blending**
  - 简单羽化融合（feather blending）
  - 接缝线优化融合（seam-aware blending）
- **Illumination compensation**
  - 全局比例光照补偿
  - 基于网格 surface 的局部光照补偿
- **Modular design**
  - 几何、特征、光照、融合模块解耦，便于扩展到视频拼接管线

---

## Project structure

```text
.
├── main.py              # 主示例：多路图像 + 柱面投影 + 接缝融合 + 网格光照
├── test.py              # 调试脚本：几何 + 光照补偿对比
├── features/            # SIFT 特征、匹配、配置类型等
│   ├── detector.py
│   ├── matcher.py
│   └── types.py
├── geometry/            # 柱面投影、单应估计、世界画布、warp 等几何模块
│   ├── canvas.py
│   ├── cylindrical.py
│   ├── homography.py
│   └── warp.py
├── rendering/           # 简单融合、接缝融合、后处理
│   ├── illumination.py
│   ├── postprocess.py
│   └── seam_blending.py
├── views/               # 视图封装 / 相机视角相关类型
│   ├── init_views.py
│   └── types.py
├── image/               # 示例图片（city/weir 等测试数据）
└── requirements.txt     # 依赖列表
```
## Installation

```bash
conda create -n test python=3.11
conda activate test

pip install -r requirements.txt

