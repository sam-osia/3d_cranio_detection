import binvox_rw as binvox_rw
import os

import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import cv2 as cv

import sys
sys.path.insert(0, '..')
from utils.utils import *

set_path()

raw_data_dir = './data/raw'
processed_data_dir = './data/processed'

display = False
save = True

AXIS_SIDE = 0
AXIS_TOP = 1
AXIS_FRONT = 2

axes = [AXIS_SIDE, AXIS_TOP, AXIS_FRONT]

for file_name in os.listdir(raw_data_dir):
    print('Processing file:', file_name)
    target_dir = os.path.join(processed_data_dir, file_name.split('.')[0])
    mkdir(target_dir)

    data = load_binvox(os.path.join(raw_data_dir, file_name))
    # data = multi_dim_padding(data, (500, 500, 500))

    for axis in axes:
        density = get_density(data, axis)
        depth = get_depths(data, axis)
        cv.imwrite(os.path.join(target_dir, f'test_depth_map_{axis}.png'), depth)

        plt.subplot(121), plt.imshow(density, cmap='gray'), plt.title('Density')
        plt.subplot(122), plt.imshow(depth, cmap='gray'), plt.title('Depth')
        if save:
            plt.savefig(os.path.join(target_dir, f'2D_map_axis{axis}.png'))
        if display:
            plt.show()

        plt.clf()

    contour = get_contour(data, axis=AXIS_SIDE)
    depth_2d = np.sum(contour, axis=1) / 255.0
    lower_bound = np.min(np.nonzero(depth_2d))
    upper_bound = np.max(np.nonzero(depth_2d))
    cutoff_neck = lower_bound + np.argmin(depth_2d[lower_bound:upper_bound - 20])
    cutoff_top = upper_bound

    plt.subplot(121), plt.imshow(contour, cmap='gray'), plt.title('Side Contour')
    plt.axhline(cutoff_neck, color='r')
    plt.subplot(122), plt.plot(depth_2d), plt.title('depth 2D')
    plt.axvline(cutoff_neck, color='r')

    if save:
        plt.savefig(os.path.join(target_dir, f'neck_cutoff.png'))
    if display:
        plt.show()
    plt.clf()

    data_top_cut = data[:, cutoff_neck:cutoff_top, :]
    contour_top_cut = get_contour(data_top_cut, axis=AXIS_SIDE)
    depth_2d = np.sum(contour_top_cut, axis=0) / 255.0
    cutoff_back = np.argmax(depth_2d > 0) - 1
    cutoff_face = 255 - np.argmax(depth_2d[::-1] > 0) + 1

    data_top_side_cut = data_top_cut[:, :, cutoff_back:cutoff_face]
    contour_top_side_cut = get_contour(data_top_side_cut, axis=AXIS_FRONT)
    depth_2d = np.sum(contour_top_side_cut, axis=1) / 255.0
    cutoff_left = np.argmax(depth_2d > 0) - 1
    cutoff_right = 255 - np.argmax(depth_2d[::-1] > 0) + 1

    data_top_side_front_cut = data_top_side_cut[cutoff_left:cutoff_right, :, :]

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(data_top_side_front_cut)

    if save:
        plt.savefig(os.path.join(target_dir, '3d_plot.png'))
    if display:
        plt.show()
    plt.clf()


