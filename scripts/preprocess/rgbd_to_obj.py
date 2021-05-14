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


def clean_mesh(mesh):
    mesh = mesh.simplify_quadric_decimation(100000)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    return mesh

set_path()

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

intrinsic = o3d.camera.PinholeCameraIntrinsic()
intrinsic.set_intrinsics(640, 480, 543.5394, 543.7316, 316.1122, 240.54805)

parent_dir = './data/examples/rgbd-scenes/s_run_104/'

frame_num = 1
color_raw = o3d.io.read_image(os.path.join(parent_dir, f'confidence_{frame_num}.png'))
depth_raw = o3d.io.read_image(os.path.join(parent_dir, f'depth_{frame_num}.png'))

rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color_raw, depth_raw)

pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image,
    intrinsic)

# Flip it, otherwise the pointcloud will be upside down
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=3),
                     fast_normal_computation=False)

distances = pcd.compute_nearest_neighbor_distance()
avg_dist = np.mean(distances)
radius = 3 * avg_dist

bpa_mesh = o3d.geometry.TriangleMesh.\
    create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector([radius, radius * 2]))

poisson_mesh = o3d.geometry.TriangleMesh.\
    create_from_point_cloud_poisson(pcd, depth=8, width=0, scale=1.1, linear_fit=False)[0]


bpa_mesh_clean = clean_mesh(bpa_mesh)
poisson_mesh_clean = clean_mesh(poisson_mesh)

o3d.io.write_point_cloud(f'./data/examples/rgbd-scenes/s_run_104/a_pcd_{frame_num}.pcd', pcd)
o3d.io.write_triangle_mesh(f'./data/examples/rgbd-scenes/s_run_104/a_bpa_mesh_{frame_num}.obj', bpa_mesh_clean)

o3d.io.write_triangle_mesh(f'./data/examples/rgbd-scenes/s_run_104/a_poisson_mesh_{frame_num}.obj', poisson_mesh_clean)

vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(pcd)
# o3d.visualization.ViewControl.set_zoom(vis.get_view_control(), 0.8)
vis.run()
