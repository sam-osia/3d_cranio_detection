import numpy as np
import binvox_rw
import open3d as o3d

import sys
sys.path.insert(0, '..')
from utils.utils import *
import matplotlib.pyplot as plt


set_path()
for i in range(1, 11):
    color_raw = o3d.io.read_image(f'./data/examples/rgbd-scenes/desk/desk_1/desk_1_{i}.png')
    depth_raw = o3d.io.read_image(f'./data/examples/rgbd-scenes/desk/desk_1/desk_1_{i}_depth.png')

    print(color_raw)
    print(depth_raw)

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    # Flip it, otherwise the pointcloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    o3d.visualization.ViewControl.set_zoom(vis.get_view_control(), 0.8)
    vis.run()
