"""SGM Stereo Matching Pipeline for Endoscope Depth Estimation."""

import cv2
import numpy as np
import tifffile
from pathlib import Path


#  1. Pre-processing

def preprocess(img, use_clahe=True):
    """Grayscale + optional CLAHE contrast enhancement."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if use_clahe:
        # CLAHE improves texture in low-contrast endoscope images
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    
    return gray


#  2. SGM Disparity Estimation (using OpenCV StereoSGBM)

def create_sgbm(min_disp=0, num_disp=128, block_size=3,
                P1=None, P2=None, disp12_max_diff=1,
                uniqueness_ratio=10, speckle_window_size=100,
                speckle_range=2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY):
    """Create StereoSGBM matcher with given parameters."""
    if P1 is None:
        P1 = 8 * 3 * block_size ** 2      # 3 channels → single channel, keep formula
    if P2 is None:
        P2 = 32 * 3 * block_size ** 2

    sgbm = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=P1,
        P2=P2,
        disp12MaxDiff=disp12_max_diff,
        uniquenessRatio=uniqueness_ratio,
        speckleWindowSize=speckle_window_size,
        speckleRange=speckle_range,
        mode=mode,
    )
    return sgbm


#  3. Post-processing/ Refinement

def wls_filter_disparity(img_left_gray, img_right_gray, sgbm, lmbda=8000, sigma=1.5):
    """Compute disparity with WLS filter (LR consistency + edge-preserving)."""
    disp_left_raw = sgbm.compute(img_left_gray, img_right_gray)

    right_matcher = cv2.ximgproc.createRightMatcher(sgbm)
    disp_right_raw = right_matcher.compute(img_right_gray, img_left_gray)

    wls = cv2.ximgproc.createDisparityWLSFilter(matcher_left=sgbm)
    wls.setLambda(lmbda)
    wls.setSigmaColor(sigma)

    disp_filtered_raw = wls.filter(
        disp_left_raw, img_left_gray,
        disparity_map_right=disp_right_raw
    )
    disp_filtered = disp_filtered_raw.astype(np.float32) / 16.0
    disp_filtered[disp_filtered < sgbm.getMinDisparity()] = np.nan

    return disp_filtered


def median_filter(disp, ksize=5):
    """Median filter for salt-and-pepper noise removal."""
    mask = np.isnan(disp)
    disp_tmp = disp.copy()
    disp_tmp[mask] = 0

    disp_med = cv2.medianBlur(disp_tmp.astype(np.float32), ksize)
    disp_med[mask] = np.nan
    return disp_med


def bilateral_filter(disp, d=7, sigma_color=10, sigma_space=10):
    """Bilateral filter for edge-preserving smoothing."""
    mask = np.isnan(disp)
    disp_tmp = disp.copy()
    disp_tmp[mask] = 0
    disp_bil = cv2.bilateralFilter(disp_tmp.astype(np.float32), d, sigma_color, sigma_space)
    disp_bil[mask] = np.nan
    return disp_bil


#  4. Disparity → Depth Conversion

def disparity_to_depth(disp, Q):
    """Convert disparity to 3D points using Q matrix. Returns (points_3d, depth_map)."""
    disp_clean = disp.copy()
    disp_clean[np.isnan(disp_clean)] = 0

    points_3d = cv2.reprojectImageTo3D(disp_clean, Q, handleMissingValues=True)

    # Mark invalid pixels
    invalid = np.isnan(disp) | (disp <= 0)
    points_3d[invalid] = [0, 0, 0]

    depth_map = points_3d[:, :, 2]
    depth_map[invalid] = np.nan

    return points_3d, depth_map


#  5. Visualization

def visualize_disparity(disp, title='Disparity', save_path=None):
    """Display and save colour-mapped disparity image."""
    disp_vis = disp.copy()
    disp_vis[np.isnan(disp_vis)] = 0

    # Normalize to 0-255
    d_min = np.nanmin(disp[~np.isnan(disp)]) if np.any(~np.isnan(disp)) else 0
    d_max = np.nanmax(disp[~np.isnan(disp)]) if np.any(~np.isnan(disp)) else 1
    disp_norm = ((disp_vis - d_min) / (d_max - d_min + 1e-6) * 255).astype(np.uint8)

    disp_color = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)
    # Black out invalid regions
    disp_color[np.isnan(disp)] = [0, 0, 0]

    cv2.imshow(title, disp_color)
    if save_path:
        cv2.imwrite(str(save_path), disp_color)

    return disp_color


# 6. Main Pipeline

def process_keyframe(keyframe_name, rectified_path, output_path,
                     sgbm_params=None, post_mode='wls', use_clahe=True,
                     submission_path=None, dataset_name='dataset_1'):
    """Full SGM pipeline: load -> preprocess -> SGBM -> postprocess -> depth -> save."""
    print(f"\n{'='*60}")
    print(f"  Processing: {keyframe_name}")
    print(f"{'='*60}")

    # Paths
    rect_dir   = rectified_path / keyframe_name / 'rectified'
    left_rect  = rect_dir / 'Left_Image_rectified.png'
    right_rect = rect_dir / 'Right_Image_rectified.png'
    q_matrix   = rect_dir / 'Q_matrix.npy'

    # Check files
    for p, label in [(left_rect, 'Left rectified'),
                     (right_rect, 'Right rectified'),
                     (q_matrix, 'Q matrix')]:
        if not p.exists():
            print(f"  [ERROR] {label} not found: {p}")
            return

    # Step 1: Load rectified images + Q matrix
    img_left  = cv2.imread(str(left_rect))
    img_right = cv2.imread(str(right_rect))
    Q = np.load(str(q_matrix))
    print(f"  Image size: {img_left.shape[1]}×{img_left.shape[0]}")

    # Step 2: Pre-processing
    gray_left  = preprocess(img_left, use_clahe=use_clahe)
    gray_right = preprocess(img_right, use_clahe=use_clahe)
    print(f"  Preprocessing: CLAHE={'ON' if use_clahe else 'OFF'})")

    # Step 3: SGM disparity estimation
    params = sgbm_params or {}
    sgbm = create_sgbm(**params)
    print(f"  SGBM params: numDisp={sgbm.getNumDisparities()}")

    # Step 4: Post-processing
    print(f"  Post-processing mode: {post_mode}")
    
    # Always apply WLS first
    disp_wls = wls_filter_disparity(gray_left, gray_right, sgbm)

    if post_mode == 'median':
        disp = median_filter(disp_wls, ksize=5)
        print("    -> WLS + Median filter")
    elif post_mode == 'bilateral':
        disp = bilateral_filter(disp_wls, d=7, sigma_color=10, sigma_space=10)
        print("    -> WLS + Bilateral filter")
    else:  # 'wls'
        disp = disp_wls
        print("    -> WLS filter only")

    valid_ratio = np.sum(~np.isnan(disp)) / disp.size * 100
    print(f"  Valid disparity: {valid_ratio:.1f}% of pixels")

    # Step 5: Disparity → Depth
    points_3d, depth_map = disparity_to_depth(disp, Q)

    # Step 6: Save results
    out_dir = output_path / keyframe_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save .npy (for internal reuse)
    np.save(str(out_dir / 'disparity.npy'), disp)
    np.save(str(out_dir / 'depth_map.npy'), depth_map)
    np.save(str(out_dir / 'points_3d.npy'), points_3d)
    visualize_disparity(disp, title=f'Disparity - {keyframe_name}',
                        save_path=out_dir / 'disparity_color.png')
    print(f"  NPY saved to: {out_dir}")

    # Save .tiff in submission structure (single-channel depth Z)
    if submission_path is not None:
        sub_dir = submission_path / dataset_name / keyframe_name
        sub_dir.mkdir(parents=True, exist_ok=True)

        depth_tiff = depth_map.astype(np.float32).copy()
        depth_tiff[np.isnan(depth_tiff) | (depth_tiff <= 0)] = 0

        tiff_path = sub_dir / 'frame1.tiff'
        tifffile.imwrite(str(tiff_path), depth_tiff)
        print(f"  TIFF saved to: {tiff_path}")

    return disp, depth_map, points_3d


def main():
    rectified_path  = Path(r'd:\FAU\Kurse\Masterarbeit\2_methods\dataset\dataset_1_rectified')
    output_path     = Path(r'd:\FAU\Kurse\Masterarbeit\2_methods\dataset\dataset_1_sgm')
    submission_path = Path(r'd:\FAU\Kurse\Masterarbeit\2_methods\submissions\sgm_submission')
    dataset_name    = 'dataset_1'

    sgbm_params = {
        'min_disp':             0,
        'num_disp':             128,    # search range; increase if close objects
        'block_size':           3,      # odd number
        'P1':                   None,   
        'P2':                   None,   
        'disp12_max_diff':      1,      # LR consistency (built-in)
        'uniqueness_ratio':     10,
        'speckle_window_size':  100,
        'speckle_range':        2,
        'mode':                 cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    }
    POST_MODE = 'wls'    # 'wls', 'median', 'bilateral'
    USE_CLAHE = True

    keyframes = sorted([d.name for d in rectified_path.iterdir()
                        if d.is_dir() and d.name.startswith('keyframe')])

    print(f"Found {len(keyframes)} keyframes: {keyframes}")

    for kf in keyframes:
        process_keyframe(
            keyframe_name=kf,
            rectified_path=rectified_path,
            output_path=output_path,
            sgbm_params=sgbm_params,
            post_mode=POST_MODE,
            use_clahe=USE_CLAHE,
            submission_path=submission_path,
            dataset_name=dataset_name
        )

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("\n✓ Done.")


if __name__ == '__main__':
    main()
