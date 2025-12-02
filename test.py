"""
test.py

静态图片几何 + 光照补偿调试脚本：
- 不做融合，仅简单覆盖
- 对比三种光照补偿：无 / 全局比例 / 本例程改进的 网格surface
"""
import numpy as np
import os
import cv2
import argparse
from pathlib import Path

from features.types import FeatureConfig
from views.init_views import init_views_from_paths
from geometry.warp import  warp_image_and_mask
from geometry.canvas import compute_panorama_canvas
from rendering.postprocess import auto_crop_valid
from rendering.illumination import (
    apply_gain_map_color,
    compute_global_gain_scalar,
    compute_grid_gain_surface,
)


def load_image_paths(folder: str):
    """
    从指定文件夹中加载所有 jpg/png/jpeg 图片路径，并按文件名排序。
    """
    folder_path = Path(folder)
    if not folder_path.is_dir():
        raise ValueError(f"输入的文件夹不存在: {folder}")

    img_paths = []
    # 顺便把原来的 'jepg' 拼写错误改成 'jpeg'
    for ext in ("*.jpg", "*.png", "*.jpeg"):
        img_paths.extend(folder_path.glob(ext))

    img_paths = sorted(str(p) for p in img_paths)
    if len(img_paths) < 2:
        raise ValueError(f"至少需要两张图片, 当前文件夹 {folder} 中只有 {len(img_paths)} 张")

    return img_paths


def parse_args():
    parser = argparse.ArgumentParser(
        description="静态图片几何 + 光照补偿调试脚本"
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default="./image/test",
        help="输入图片所在文件夹，默认 ./image/2",
    )
    return parser.parse_args()


def build_src_mask(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return (img.sum(axis=2) > 0).astype(np.uint8)
    return (img > 0).astype(np.uint8)

def main(folder: str):
    # 1. 加载图片路径
    img_paths = load_image_paths(folder)

    cfg = FeatureConfig(
        use_cylindrical=False,  # 柱面投影(True：开启)
        cyl_f_ratio=0.8,        # 参数
    )  # 特征提取/匹配配置

    # 统一在这里指定参考视图是哪一张：取中间那张
    ref_id = len(img_paths) // 2

    # 1. 初始化视图（读图、特征、匹配、H_to_ref 都在里面完成）
    views = init_views_from_paths(img_paths, cfg, ref_id=ref_id)

    # 2. 选出参考视图，拿到 ref 的尺寸
    ref_view = next(v for v in views if v.id == ref_id)
    h_ref, w_ref = ref_view.image.shape[:2]

    # ------------------------------------------------------------------
    # B. 世界画布：把所有视图统一到 global canvas 上
    # ------------------------------------------------------------------
    # 1) 准备 images 和 H_to_ref 列表
    images = [v.image for v in views]

    H_to_ref = []
    for v in views:
        if v.id == ref_id:
            # 参考视图自己：image -> ref 就是单位阵
            H_to_ref.append(np.eye(3, dtype=np.float32))
        else:
            # 其它视图：用 init_views_from_paths 里算好的 H_to_ref
            H_to_ref.append(v.H_to_ref.astype(np.float32))

    # 2) 计算世界画布几何（全局 min/max + 平移矩阵 T）
    pano_h, pano_w, T = compute_panorama_canvas(
        images=images,
        H_to_ref=H_to_ref,
        ref_size=(h_ref, w_ref),
    )
    print("world canvas size:", pano_w, pano_h)
    print("T:\n", T)

    # 3) 每张图到世界画布的单应矩阵 H_to_canvas = T @ H_to_ref
    H_to_canvas = [T @ H for H in H_to_ref]
    os.makedirs("debug_canvas", exist_ok=True)

    # 4) warp 到世界画布，并分别保存
    world_warp_list = []
    world_masks = []

    for v, img, Hc in zip(views, images, H_to_canvas):
        src_mask = build_src_mask(img)
        warped_canvas, mask_valid = warp_image_and_mask(
            img,
            Hc,
            (pano_h, pano_w),
            src_mask=src_mask,
        )
        world_warp_list.append(warped_canvas)
        world_masks.append(mask_valid)

        cv2.imwrite(
            os.path.join("debug_canvas", f"view{v.id}_on_canvas.jpg"),
            warped_canvas,
        )

    # ------------------------------------------------------------------
    # 简单覆盖 + 裁剪 
    # ------------------------------------------------------------------
    def save_pano_for_world_warps(world_warps, suffix: str):
        pano_simple = np.zeros_like(world_warps[0])

        # 先放参考视图
        pano_simple[...] = world_warps[ref_id]

        # 其它视图覆盖上去（简单覆盖，不做融合）
        for i, warped in enumerate(world_warps):
            if i == ref_id:
                continue
            mask = warped.sum(axis=2) > 0
            pano_simple[mask] = warped[mask]

        pano_crop, bbox = auto_crop_valid(pano_simple, margin=10)
        crop_path = os.path.join(
            "debug_canvas", f"pano_simple_crop_{suffix}.jpg"
        )
        cv2.imwrite(crop_path, pano_crop)
        print(f"[{suffix}] 已保存裁剪画布: {crop_path}, bbox={bbox}")

    # ================================================================
    # 不做光照补偿（None）
    # ================================================================
    save_pano_for_world_warps(world_warp_list, suffix="none")

    # ================================================================
    # 整体比例法（global gain）
    # ================================================================
    world_warp_global = [img.copy() for img in world_warp_list]
    ref_img = world_warp_global[ref_id]
    ref_mask = world_masks[ref_id]

    for i in range(len(world_warp_global)):
        if i == ref_id:
            continue

        overlap = (ref_mask > 0) & (world_masks[i] > 0)
        g = compute_global_gain_scalar(ref_img, world_warp_global[i], overlap)

        h, w = ref_mask.shape
        gain_map = np.ones((h, w, 1), dtype=np.float32)
        gain_map[:, :, 0] = g

        world_warp_global[i] = apply_gain_map_color(
            world_warp_global[i],
            gain_map,
        )

    save_pano_for_world_warps(world_warp_global, suffix="global")

    # =========================
    # 网格 Surface 改进法
    # =========================
    world_warp_grid = [img.copy() for img in world_warp_list]
    ref_img = world_warp_grid[ref_id]
    ref_mask = world_masks[ref_id]

    for i in range(len(world_warp_grid)):
        if i == ref_id:
            continue

        tgt_before = world_warp_list[i]
        tgt_mask = world_masks[i]

        gain_map = compute_grid_gain_surface(
            ref_img, tgt_before, ref_mask, tgt_mask
        )
        world_warp_grid[i] = apply_gain_map_color(tgt_before, gain_map)

    save_pano_for_world_warps(world_warp_grid, suffix="grid_surface")

if __name__ == "__main__":
    args = parse_args()
    main(args.folder)
