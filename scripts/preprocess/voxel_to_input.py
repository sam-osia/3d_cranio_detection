import cv2

import binvox_rw as binvox_rw
import os

import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import cv2 as cv

import sys
sys.path.insert(0, '..')
from utils.utils import *

set_path()


def find_bounds(depth_map, search_axis):
    if search_axis == 0:
        empty_rows = [0, depth_map.shape[search_axis] - 1]
        for i in range(depth_map.shape[search_axis]):
            row = depth_map[i, :]
            point_count = np.count_nonzero(row)
            if point_count == 0:
                empty_rows.append(i)

        empty_rows = np.array(empty_rows)
        empty_rows_lower = empty_rows[empty_rows < int(depth_map.shape[search_axis] / 2)]
        empty_rows_upper = empty_rows[empty_rows > int(depth_map.shape[search_axis] / 2)]

        lower_bound = np.max(empty_rows_lower)
        upper_bound = np.min(empty_rows_upper)
        return lower_bound, upper_bound
    elif search_axis == 1:
        empty_rows = [0, depth_map.shape[search_axis] - 1]
        for i in range(depth_map.shape[search_axis]):
            row = depth_map[:, i]
            point_count = np.count_nonzero(row)
            if point_count == 0:
                empty_rows.append(i)

        empty_rows = np.array(empty_rows)
        empty_rows_lower = empty_rows[empty_rows < int(depth_map.shape[search_axis] / 2)]
        empty_rows_upper = empty_rows[empty_rows > int(depth_map.shape[search_axis] / 2)]

        lower_bound = np.max(empty_rows_lower)
        upper_bound = np.min(empty_rows_upper)
        return lower_bound, upper_bound


def pad_to_square(depth_map, pad_with=0, extra_padding=0):
    (a, b) = depth_map.shape
    if a > b:
        padding_left = padding_right = int((a - b) / 2)
        if (a - b) % 2 == 1:
            padding_left += 1
        padding = ((extra_padding, extra_padding), (padding_left + extra_padding, padding_right + extra_padding))
    else:
        padding_top = padding_bottom = int((b - a) / 2)
        if (b - a) % 2 == 1:
            padding_top += 1
        padding = ((padding_top + extra_padding, padding_bottom + extra_padding), (extra_padding, extra_padding))
    return np.pad(depth_map, padding, mode='constant', constant_values=pad_with)


def normalize_depths(depth_map, invert=False):
    max_val = np.max(depth_map)
    min_val = np.min(depth_map[depth_map != 0])

    depth_shape = depth_map.shape
    zero_mask = depth_map == 0
    depth_map = depth_map.reshape(-1)
    depth_map = depth_map - min_val
    depth_map = depth_map / (max_val - min_val)
    if invert:
        depth_map = 1 - depth_map

    depth_map = np.reshape(depth_map, depth_shape)
    depth_map[zero_mask] = 0

    return depth_map


raw_data_parent = './data/raw/3dmd_voxels'
processed_data_parent = './data/processed/voxel'

for file_name in os.listdir(raw_data_parent):
    if 'white_with_cap' not in file_name:
        continue

    processed_data_dir = os.path.join(processed_data_parent, file_name.split('.')[0])
    mkdir(processed_data_dir)

    data = load_binvox(os.path.join(raw_data_parent, file_name))

    density = np.sum(data[:, 175:, :], axis=1)

    h_lower, h_upper = find_bounds(density, search_axis=0)
    v_lower, v_upper = find_bounds(density, search_axis=1)
    depth_trimmed = density[h_lower:h_upper, v_lower:v_upper]
    depth_padded = pad_to_square(depth_trimmed, extra_padding=0)
    depth_normalized = normalize_depths(depth_padded, invert=False)
    depth_resized = cv2.resize(depth_normalized, dsize=(128, 128), interpolation=cv2.INTER_LINEAR)

    np.save(os.path.join(processed_data_dir, 'depths'), depth_resized)

    plt.suptitle(file_name)

    plt.subplot(231), plt.imshow(density), plt.title('find bounds')
    plt.hlines(h_lower, 0, density.shape[1] - 1, colors='red')
    plt.hlines(h_upper, 0, density.shape[1] - 1, colors='red')
    plt.vlines(v_lower, 0, density.shape[0] - 1, colors='red')
    plt.vlines(v_upper, 0, density.shape[0] - 1, colors='red')

    plt.subplot(232), plt.imshow(depth_trimmed), plt.title('trim to edge')
    plt.subplot(233), plt.imshow(depth_padded), plt.title('pad to square')
    plt.subplot(234), plt.imshow(depth_normalized), plt.title('normalize depths')
    plt.subplot(235), plt.imshow(depth_resized), plt.title('resize to 128x128')

    plt.show()


