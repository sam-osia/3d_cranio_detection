import numpy as np
import binvox_rw
import open3d as o3d
from PIL import Image
import time
import sys
sys.path.insert(0, '..')
from utils.utils import *
import matplotlib.pyplot as plt
import cv2
import copy


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


set_path()

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

intrinsic = o3d.camera.PinholeCameraIntrinsic()
intrinsic.set_intrinsics(640, 480, 543.5394, 543.7316, 316.1122, 240.54805)


parent_dir = './data/examples/rgbd-scenes/s_run_104/'

voxel_size = 0.006

prev_trans = [[0.862, 0.011, -0.507, 0.0], [-0.139, 0.967, -0.215, 0.2],
         [0.487, 0.255, 0.835, -0.4], [0.0, 0.0, 0.0, 1.0]]
# prev_trans = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

prev_pcd = o3d.io.read_point_cloud(os.path.join(parent_dir, f'pcd_{1}.pcd'))

for i in range(2, 261):
    target_raw = o3d.io.read_point_cloud(os.path.join(parent_dir, f'pcd_{i}.pcd'))
    source = prev_pcd.voxel_down_sample(voxel_size=voxel_size)
    target = target_raw.voxel_down_sample(voxel_size=voxel_size)

    # source = prev_pcd
    # target = target_raw

    source.transform(prev_trans)

    source.paint_uniform_color([1, 0, 0])
    target.paint_uniform_color([0, 0, 1])

    vis = o3d.visualization.Visualizer()
    vis.create_window('samsung_run_104')
    vis.add_geometry(source)
    vis.add_geometry(target)
    threshold = 0.3
    icp_iteration = 100
    save_image = False

    # time.sleep(1)

    for i in range(icp_iteration):
        reg_p2l = o3d.pipelines.registration.registration_icp(
            source, target, threshold, np.identity(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1))

        prev_trans = np.matmul(prev_trans, reg_p2l.transformation)
        source.transform(reg_p2l.transformation)
        vis.update_geometry(source)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.2)
        if save_image:
            vis.capture_screen_image("temp_%04d.jpg" % i)

    prev_pcd = source + target
    prev_pcd.voxel_down_sample(voxel_size=voxel_size)
    print(prev_trans)
    time.sleep(0.5)
    # vis.destroy_window()
    break



exit()

for i in range(2, 263):
    source = o3d.io.read_point_cloud(os.path.join(parent_dir, f'pcd_{i - 1}.pcd'))
    target = o3d.io.read_point_cloud(os.path.join(parent_dir, f'pcd_{i}.pcd'))
    threshold = 0.2

    trans_init = np.asarray([[0.862, 0.011, -0.507, 0.5],
                             [-0.139, 0.967, -0.215, 0.7],
                             [0.487, 0.255, 0.835, -1.4], [0.0, 0.0, 0.0, 1.0]])

    draw_registration_result(source, target, trans_init)


# frame_num = 200
# color_raw = o3d.io.read_image(os.path.join(parent_dir, f'confidence_{frame_num}.png'))
# depth_raw = o3d.io.read_image(os.path.join(parent_dir, f'depth_{frame_num}.png'))
#
# print(color_raw)
# print(depth_raw)
#
# rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
#     color_raw, depth_raw)
#
# pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
#     rgbd_image,
#     intrinsic)
#
# # Flip it, otherwise the pointcloud will be upside down
# pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
#
# # o3d.io.write_point_cloud(f'./data/examples/rgbd-scenes/s_run_104/pcd_{i}.pcd', pcd)
# vis = o3d.visualization.Visualizer()
# vis.create_window()
# vis.add_geometry(pcd)
# o3d.visualization.ViewControl.set_zoom(vis.get_view_control(), 0.8)
# vis.run()
