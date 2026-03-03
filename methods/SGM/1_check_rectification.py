"""
- Epipolar lines visualization
- Y-disparity histogram (use |mean| to judge quality)
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def draw_epipolar_lines(img_left, img_right, num_lines=20):
    """Draw horizontal lines across the concatenated stereo pair."""
    h, w = img_left.shape[:2]
    canvas = np.concatenate([img_left, img_right], axis=1)

    step = h // (num_lines + 1)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

    for i in range(1, num_lines + 1):
        y = i * step
        color = colors[i % len(colors)]
        cv2.line(canvas, (0, y), (canvas.shape[1], y), color, 1, cv2.LINE_AA)

    return canvas


def compute_y_disparity(img_left, img_right, max_features=500):
    """
    Compute y-disparity using ORB feature matching.
    Returns array of y-coordinate differences for matched points.
    """
    gray_l = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=max_features)
    kp1, des1 = orb.detectAndCompute(gray_l, None)
    kp2, des2 = orb.detectAndCompute(gray_r, None)

    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(good) < 5:
        return None

    y_disparities = []
    for m in good:
        y1 = kp1[m.queryIdx].pt[1]
        y2 = kp2[m.trainIdx].pt[1]
        y_disparities.append(y1 - y2)

    return np.array(y_disparities)


def check_rectification(keyframe_name, img_left, img_right, mean_threshold=2.0):
    """
    Check rectification quality for one keyframe.
    
    Args:
        mean_threshold: |mean| < threshold means GOOD (default: 2.0 pixels)
    """
    print(f"\n{'='*50}")
    print(f"  {keyframe_name}")
    print(f"{'='*50}")

    # 1) Epipolar lines
    epipolar_canvas = draw_epipolar_lines(img_left, img_right, num_lines=20)

    # 2) Y-disparity
    y_disp = compute_y_disparity(img_left, img_right)

    if y_disp is not None:
        mean_val = np.mean(y_disp)
        std_val = np.std(y_disp)
        ok = abs(mean_val) < mean_threshold
        
        print(f"  Matches: {len(y_disp)}")
        print(f"  Y-disp mean: {mean_val:.2f} px")
        print(f"  Y-disp std:  {std_val:.2f} px")
        print(f"  Result: {'GOOD' if ok else 'BAD'} (|mean| {'<' if ok else '>='} {mean_threshold} px)")
    else:
        ok = False
        mean_val = None
        print("  Not enough features for y-disparity check")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    status = "GOOD" if ok else "BAD"
    fig.suptitle(f"{keyframe_name} - {status}", fontsize=14)

    # Left: epipolar lines
    axes[0].imshow(cv2.cvtColor(epipolar_canvas, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Epipolar Lines")
    axes[0].axis('off')

    # Right: y-disparity histogram
    if y_disp is not None:
        axes[1].hist(y_disp, bins=30, color='steelblue', edgecolor='black')
        axes[1].axvline(0, color='red', linestyle='--', linewidth=2, label='ideal (0)')
        axes[1].axvline(mean_val, color='orange', linestyle='-', linewidth=2, label=f'mean ({mean_val:.2f})')
        axes[1].set_xlabel("Y-Disparity (px)")
        axes[1].set_ylabel("Count")
        axes[1].set_title(f"Y-Disparity (mean={mean_val:.2f} px)")
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, "N/A", ha='center', va='center', fontsize=20)
        axes[1].set_title("Y-Disparity")

    plt.tight_layout()
    plt.show()

    return ok


def main():
    rectified_base = Path(r'd:\FAU\Kurse\Masterarbeit\2_methods\dataset\dataset_1_rectified')

    keyframes = sorted(rectified_base.glob('keyframe_*'))
    if not keyframes:
        print("No rectified keyframe folders found!")
        return

    results = []
    for kf in keyframes:
        left_path = kf / 'rectified' / 'Left_Image_rectified.png'
        right_path = kf / 'rectified' / 'Right_Image_rectified.png'

        if not left_path.exists() or not right_path.exists():
            print(f"Skipping {kf.name}: images not found")
            continue

        img_left = cv2.imread(str(left_path))
        img_right = cv2.imread(str(right_path))

        if img_left is None or img_right is None:
            print(f"Skipping {kf.name}: failed to load")
            continue

        ok = check_rectification(kf.name, img_left, img_right)
        results.append((kf.name, ok))

    # Summary
    print(f"\n{'='*50}")
    print("  SUMMARY")
    print(f"{'='*50}")
    for name, ok in results:
        print(f"  {name}: {'GOOD' if ok else 'BAD'}")


if __name__ == '__main__':
    main()
