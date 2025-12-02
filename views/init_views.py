from __future__ import annotations

from typing import Tuple, Optional, List
import cv2
import numpy as np

from features.types import FeatureConfig
from features.detector import extract_sift_features
from features.matcher import match_chain_sequential
from geometry.homography import estimate_homography_ransac
from geometry.cylindrical import cylindrical_project
from .types import View

def init_views_from_paths(
    img_paths: List[str],
    cfg: FeatureConfig,
    ref_id: int = 0,
) -> List[View]:
    """
    通用版本：从若干张图片路径初始化一组 View，并计算每张图到参考视图的单应矩阵 H_to_ref。

    Parameters
    ----------
    img_paths : List[str]
        多张图片的路径，建议按空间顺序（如从左到右）排列。
    cfg : FeatureConfig
        特征提取 / 匹配配置。
    ref_id : int
        参考视图的索引（0 <= ref_id < len(img_paths)）。

    Returns
    -------
    views : List[View]
        views[i].id      = i
        views[i].H_to_ref: image i -> ref_id 的平面坐标系
                           （ref 视图自身为单位阵）
    """
    num_imgs = len(img_paths)
    if num_imgs < 2:
        raise ValueError(f"至少需要两张图片进行拼接，当前只有 {num_imgs} 张")

    if not (0 <= ref_id < num_imgs):
        raise ValueError(f"ref_id 必须在 [0, {num_imgs - 1}] 范围内，当前为 {ref_id}")

    # ------------------------------------------------------------------
    # 1. 读图
    # ------------------------------------------------------------------
    images: List[np.ndarray] = []
    for p in img_paths:
        img = cv2.imread(p)
        if img is None:
            raise IOError(f"读图失败: {p}")
        images.append(img)
    # ------------------------------------------------------------------
    # 1.5 柱面投影(CP-SIFT)
    # ------------------------------------------------------------------
    proc_images: List[np.ndarray] = []

    for img in images:
        proc = img

        if cfg.use_cylindrical:
            h, w = img.shape[:2]

            if cfg.fx is not None and cfg.fy is not None:
                fx = cfg.fx
                fy = cfg.fy
            else:
                f = cfg.cyl_f_ratio * w
                fx = fy = float(f)

            cx = cfg.cx if cfg.cx is not None else w * 0.5
            cy = cfg.cy if cfg.cy is not None else h * 0.5

            proc, _ = cylindrical_project(img, fx, fy, cx, cy)

        proc_images.append(proc)

    # ------------------------------------------------------------------
    # 2. 提取特征（每张图一个 FeatureSet）
    # ------------------------------------------------------------------
    features = []
    for i, img in enumerate(proc_images):
        f = extract_sift_features(img, image_id=i, cfg=cfg)
        features.append(f)

    # ------------------------------------------------------------------
    # 3. 顺序匹配：0-1, 1-2, ..., (N-2)-(N-1)
    #    match_chain_sequential 返回的长度应为 num_imgs-1
    # ------------------------------------------------------------------
    match_results = match_chain_sequential(features, cfg)
    if len(match_results) != num_imgs - 1:
        raise RuntimeError(
            f"match_chain_sequential 结果数量异常：期望 {num_imgs - 1}，实际 {len(match_results)}"
        )

    # ------------------------------------------------------------------
    # 4. 对每一对相邻视图估计 H_forward[i] : i -> i+1
    # ------------------------------------------------------------------
    H_forward: List[np.ndarray] = []
    for i in range(num_imgs - 1):
        fi = features[i]
        fj = features[i + 1]
        mij = match_results[i]

        H_res = estimate_homography_ransac(
            fi, fj, mij,
            ransac_thresh=3.0,
            confidence=0.995,
        )
        H_ij = H_res.H.astype(np.float32)  # i -> i+1
        H_forward.append(H_ij)

    # ------------------------------------------------------------------
    # 5. 根据 ref_id 传播，得到每张图到 ref 的单应矩阵 H_to_ref[i]
    #
    #    - H_to_ref[ref_id] = I
    #    - 左侧 (i < ref_id) : H_to_ref[i]   = H_to_ref[i+1] @ H_forward[i]
    #    - 右侧 (i > ref_id) : H_to_ref[i]   = H_to_ref[i-1] @ inv(H_forward[i-1])
    # ------------------------------------------------------------------
    H_to_ref: List[np.ndarray] = [np.eye(3, dtype=np.float32) for _ in range(num_imgs)]
    H_to_ref[ref_id] = np.eye(3, dtype=np.float32)

    # 从 ref 向左传播：ref-1, ref-2, ..., 0
    for i in range(ref_id - 1, -1, -1):
        # i -> ref = (i+1 -> ref) ∘ (i -> i+1)
        H_to_ref[i] = H_to_ref[i + 1] @ H_forward[i]

    # 从 ref 向右传播：ref+1, ref+2, ..., N-1
    for i in range(ref_id + 1, num_imgs):
        # i -> ref = (i-1 -> ref) ∘ (i -> i-1)
        # 其中 i -> i-1 = inv( (i-1) -> i ) = inv(H_forward[i-1])
        H_i_to_i_1 = np.linalg.inv(H_forward[i - 1])
        H_to_ref[i] = H_to_ref[i - 1] @ H_i_to_i_1

    # ------------------------------------------------------------------
    # 6. 构造 View 列表
    # ------------------------------------------------------------------
    views: List[View] = []
    for i, (path, img, feat, Href) in enumerate(
        zip(img_paths, proc_images, features, H_to_ref)
    ):
        v = View(
            id=i,
            name=path,
            image=img,
            features=feat,
            H_to_ref=Href,
        )
        views.append(v)

    return views