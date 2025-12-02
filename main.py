"""
main.py

多视角静态图片拼接 Demo：
- 可选柱面投影
- 世界画布构建
- 羽化融合 & 接缝融合
- 网格光照补偿 (grid surface)

This script is the main static panorama demo for the project.
"""

import numpy as np
import os
import cv2
import argparse
from pathlib import Path
from typing import Optional, List

from features.types import FeatureConfig
from views.init_views import init_views_from_paths
from geometry.warp import warp_view_to_ref, warp_image_and_mask
from geometry.canvas import compute_panorama_canvas
from rendering.postprocess import auto_crop_valid
from rendering.illumination import (
    apply_gain_map_color,
    compute_grid_gain_surface,
)
from rendering.simple_blending import render_pano_feather
from rendering.seam_blending import render_pano_seam


def load_image_paths(folder: str):
    folder_path = Path(folder)
    if not folder_path.is_dir():
        raise ValueError(f"输入的文件夹不存在: {folder}")

    img_paths = []
    for ext in ("*.jpg", "*.png", "*.jpeg"):
        img_paths.extend(folder_path.glob(ext))

    img_paths = sorted(str(p) for p in img_paths)
    if len(img_paths) < 2:
        raise ValueError(f"至少需要两张图片, 当前文件夹 {folder} 中只有 {len(img_paths)} 张")

    return img_paths


def parse_args():
    parser = argparse.ArgumentParser(
        description="羽化融合 + 接缝融合 + 光照补偿 对比脚本"
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default="./image/test",
        help="输入图片所在文件夹，默认 ./image/2",
    )
    # 柱面投影开关
    parser.add_argument(
        "--cyl",
        action="store_true",
        help="(use_cylindrical=True)",
    )
    # 柱面投影焦距系数
    parser.add_argument(
        "--cyl-f",
        type=float,
        default=0.8,
        help="default 0.8",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="默认不显示",
    )

    return parser.parse_args()



def build_src_mask(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return (img.sum(axis=2) > 0).astype(np.uint8)
    return (img > 0).astype(np.uint8)

def main(
    folder: str,
    use_cylindrical: bool = False,
    cyl_f_ratio: float = 0.8,
    preview: bool = False,  
):
    img_paths = load_image_paths(folder)

    cfg = FeatureConfig(
        use_cylindrical=use_cylindrical,  # True 时开启柱面投影
        cyl_f_ratio=cyl_f_ratio,
    )

    ref_id = len(img_paths) // 2

    views = init_views_from_paths(img_paths, cfg, ref_id=ref_id)

    ref_view = next(v for v in views if v.id == ref_id)
    h_ref, w_ref = ref_view.image.shape[:2]

    # ------------------------------------------------------------------
    # A. warp 到参考画布
    # ------------------------------------------------------------------
    OUT_DIR = "results"
    os.makedirs(OUT_DIR, exist_ok=True)

    if preview:
        for v in views:
            warped = warp_view_to_ref(v, (h_ref, w_ref))

            preview_scale = 0.25
            ph = int(h_ref * preview_scale)
            pw = int(w_ref * preview_scale)
            preview_img = cv2.resize(warped, (pw, ph), interpolation=cv2.INTER_AREA)
            cv2.imshow(f"view{v.id}_preview", preview_img)

        print("按任意键关闭预览窗口，继续世界画布调试...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    # ------------------------------------------------------------------
    # B. 世界画布
    # ------------------------------------------------------------------

    images = [v.image for v in views]

    H_to_ref = []
    for v in views:
        if v.id == ref_id:
            H_to_ref.append(np.eye(3, dtype=np.float32))
        else:
            H_to_ref.append(v.H_to_ref.astype(np.float32))

    pano_h, pano_w, T = compute_panorama_canvas(
        images=images,
        H_to_ref=H_to_ref,
        ref_size=(h_ref, w_ref),
    )
    print("world canvas size:", pano_w, pano_h)
    print("T:\n", T)
    H_to_canvas = [T @ H for H in H_to_ref]

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

    # ------------------------------------------------------------------
    # C. 双融合图片生成
    # ------------------------------------------------------------------
    def save_pano_feather_for_world_warps(
        world_warps,
        suffix: str = "",
        feather_width: int = 64,
    ):
        pano_feather, pano_mask = render_pano_feather(
            world_warps,
            world_masks,
            ref_id=ref_id,
            feather_width=feather_width,
        )

        pano_crop, bbox = auto_crop_valid(pano_feather, margin=10)

        name_suffix = f"_{suffix}" if suffix else ""
        crop_path = os.path.join(OUT_DIR, f"pano_feather{name_suffix}.jpg")
        tag_suffix = f"-{suffix}" if suffix else ""
        print(f"[FEATHER{tag_suffix}] 已保存裁剪画布(结果目录): {crop_path}, bbox={bbox}")

        cv2.imwrite(crop_path, pano_crop)


    def save_pano_seam_for_world_warps(
        world_warps,
        suffix: str = "",
        band_width: int = 32,
        fg_masks: Optional[List[np.ndarray]] = None,
    ):
        pano_seam = render_pano_seam(
            world_warps,
            world_masks,
            ref_id=ref_id,
            band_width=band_width,
            forbid_masks=fg_masks,
            w_f=10.0,
        )

        pano_crop, bbox = auto_crop_valid(pano_seam, margin=10)

        name_suffix = f"_{suffix}" if suffix else ""
        crop_path = os.path.join(OUT_DIR, f"pano_seam{name_suffix}.jpg")
        tag_suffix = f"-{suffix}" if suffix else ""
        print(f"[SEAM{tag_suffix}] 已保存裁剪画布(结果目录): {crop_path}, bbox={bbox}")

        cv2.imwrite(crop_path, pano_crop)

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

    save_pano_feather_for_world_warps(world_warp_grid)
    save_pano_seam_for_world_warps(world_warp_grid)

if __name__ == "__main__":
    args = parse_args()
    main(
        folder=args.folder,
        use_cylindrical=args.cyl,
        cyl_f_ratio=args.cyl_f,
        preview=args.preview,
    )
