import sys
import numpy as np
import cv2
import open3d as o3d
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent / 'dataset'
SGM_DIR = BASE_DIR / 'dataset_1_sgm'
RECT_DIR = BASE_DIR / 'dataset_1_rectified'

keyframe = 'keyframe_1'

# Load 3D points (H x W x 3)
pts_path = SGM_DIR / keyframe / 'points_3d.npy'
print(f'Loading {pts_path} ...')
points_3d = np.load(str(pts_path))

# Load rectified left image for true colors
img_path = RECT_DIR / keyframe / 'rectified' / 'Left_Image_rectified.png'
print(f'Loading {img_path} ...')
img_left = cv2.imread(str(img_path))
img_rgb = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)  # BGR -> RGB

# Flatten to (N, 3)
H, W = points_3d.shape[:2]
pts = points_3d.reshape(-1, 3)
colors_flat = img_rgb.reshape(-1, 3).astype(np.float64) / 255.0  # normalize to [0, 1]

# Filter invalid points
valid = np.isfinite(pts).all(axis=1) & (pts[:, 2] > 0)
pts = pts[valid]
colors_flat = colors_flat[valid]
print(f'Total valid points: {len(pts):,}')

# Build point cloud with original image colors
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts)
pcd.colors = o3d.utility.Vector3dVector(colors_flat)

# Visualize
print('Controls: Left-drag=rotate, Scroll=zoom, Middle/Shift+Left=pan, R=reset view')
o3d.visualization.draw_geometries(
    [pcd],
    window_name=f'SGM Point Cloud - {keyframe}',
    width=1280, height=720,
)
