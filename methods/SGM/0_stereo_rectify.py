import cv2
import numpy as np
import yaml
import os
from pathlib import Path

def load_calibration(yaml_path):
    """Read calibration parameters."""
    fs = cv2.FileStorage(str(yaml_path), cv2.FILE_STORAGE_READ)
    
    R = fs.getNode('R').mat()      # Rotation matrix 3×3 
    T = fs.getNode('T').mat()      # Translation vector 1×3
    K1 = fs.getNode('M1').mat()    # Left camera intrinsic matrix 3×3
    D1 = fs.getNode('D1').mat()    # Left camera distortion coefficients 1×5
    K2 = fs.getNode('M2').mat()    # Right camera intrinsic matrix 3×3
    D2 = fs.getNode('D2').mat()    # Right camera distortion coefficients 1×5
    
    fs.release()
 
    T = T.reshape(3, 1)
    
    return K1, D1, K2, D2, R, T


def stereo_rectify(img_left, img_right, K1, D1, K2, D2, R, T):
    """Perform stereo rectification on a pair of images."""
    h, w = img_left.shape[:2]
    
    # Compute rectification transforms
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K1, D1, K2, D2, (w, h), R, T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0
    )   # Q: Disparity-to-depth mapping matrix, roi1, roi2: Region of Interest
    
    # Compute rectification maps (map1: x, map2: y)
    map1_left, map2_left = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (w, h), cv2.CV_32FC1)
    map1_right, map2_right = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (w, h), cv2.CV_32FC1)
    
    # Apply rectification
    img_left_rectified = cv2.remap(img_left, map1_left, map2_left, cv2.INTER_LINEAR)
    img_right_rectified = cv2.remap(img_right, map1_right, map2_right, cv2.INTER_LINEAR)
    
    return img_left_rectified, img_right_rectified, Q


def process_keyframe(keyframe_path, output_base_path):
    """Process the stereo image pair in a keyframe folder."""
    keyframe_path = Path(keyframe_path)
    calib_path = keyframe_path / 'endoscope_calibration.yaml'

    # Check calibration file exists
    if not calib_path.exists():
        print(f"Calibration file not found: {calib_path}")
        return

    # Check image files exist
    left_img_path = keyframe_path / 'Left_Image.png'
    right_img_path = keyframe_path / 'Right_Image.png'

    if not left_img_path.exists() or not right_img_path.exists():
        print(f"Image pair not found in: {keyframe_path}")
        return

    # Load calibration
    K1, D1, K2, D2, R, T = load_calibration(calib_path)

    # Load images
    img_left = cv2.imread(str(left_img_path))
    img_right = cv2.imread(str(right_img_path))

    if img_left is None or img_right is None:
        print(f"Failed to load images in: {keyframe_path}")
        return

    # Rectify
    img_left_rect, img_right_rect, Q = stereo_rectify(
        img_left, img_right, K1, D1, K2, D2, R, T
    )

    # Create output directory and save
    output_path = Path(output_base_path) / keyframe_path.name / 'rectified'
    output_path.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_path / 'Left_Image_rectified.png'), img_left_rect)
    cv2.imwrite(str(output_path / 'Right_Image_rectified.png'), img_right_rect)
    
    # Save Q matrix for SGM disparity-to-depth conversion
    np.save(str(output_path / 'Q_matrix.npy'), Q)

    print(f"Processed: {keyframe_path.name} (Q matrix saved)")


def main():
    dataset_path = Path(r'd:\FAU\Kurse\Masterarbeit\2_methods\dataset\dataset_1')
    output_path = Path(r'd:\FAU\Kurse\Masterarbeit\2_methods\dataset\dataset_1_rectified')
    
    # Process each keyframe folder
    keyframes = sorted(dataset_path.glob('keyframe_*'))
    
    for keyframe in keyframes:
        print(f"\nProcessing {keyframe.name}...")
        process_keyframe(keyframe, output_path)
    
    print("\nStereo rectification completed!")


if __name__ == '__main__':
    main()