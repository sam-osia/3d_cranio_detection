import imageio
import matplotlib.pyplot as plt
import cv2

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


def normalize_depths(depth_map, invert=False, offset=0, force_zero=False):
    depth_shape = depth_map.shape
    zero_mask = depth_map == 0
    depth_map = depth_map.reshape(-1)

    depth_map = depth_map + offset

    max_val = np.max(depth_map)
    min_val = np.min(depth_map[depth_map != offset]) - offset
    print(max_val, min_val)

    if force_zero:
        depth_map = (depth_map - min_val)
        depth_map = depth_map / (max_val - min_val)
    else:
        depth_map = depth_map / max_val

    if invert:
        depth_map = 1 - depth_map

    depth_map = np.reshape(depth_map, depth_shape)
    depth_map[zero_mask] = 0

    return depth_map


raw_data_parent = './data/raw/rgbd'
processed_data_parent = './data/processed/rgbd'
run_id = 's_1_top'

raw_data_dir = os.path.join(raw_data_parent, run_id)
processed_data_dir = os.path.join(processed_data_parent, run_id)
mkdir(processed_data_dir)

for frame_ind in range(2, 100):
    rgb_file = imageio.imread(os.path.join(raw_data_dir, f'rgb_{frame_ind}.png'))
    mask_file = imageio.imread(os.path.join(raw_data_dir, f'confidence_{frame_ind}.png'))
    depth_file = imageio.imread(os.path.join(raw_data_dir, f'depth_{frame_ind}.png'))

    # depth_file = cv2.rotate(depth_file, cv2.ROTATE_180)

    mask_file = np.array(mask_file)
    depth_file = np.array(depth_file)

    h_lower, h_upper = find_bounds(depth_file, search_axis=0)
    v_lower, v_upper = find_bounds(depth_file, search_axis=1)
    depth_trimmed = depth_file[h_lower:h_upper, v_lower:v_upper]
    depth_padded = pad_to_square(depth_trimmed)
    depth_normalized = normalize_depths(depth_padded, invert=True, force_zero=True, offset=0)
    depth_resized = cv2.resize(depth_normalized, dsize=(128, 128), interpolation=cv2.INTER_LINEAR)

    np.save(os.path.join(processed_data_dir, 'depths'), depth_resized)

    plt.subplot(231), plt.imshow(rgb_file), plt.title('rgb')

    plt.subplot(232), plt.imshow(depth_file), plt.title('find bounds')
    plt.hlines(h_lower, 0, depth_file.shape[1] - 1, colors='red')
    plt.hlines(h_upper, 0, depth_file.shape[1] - 1, colors='red')
    plt.vlines(v_lower, 0, depth_file.shape[0] - 1, colors='red')
    plt.vlines(v_upper, 0, depth_file.shape[0] - 1, colors='red')

    plt.subplot(233), plt.imshow(depth_trimmed), plt.title('trim to edge')
    plt.subplot(234), plt.imshow(depth_padded), plt.title('pad to square')
    plt.subplot(235), plt.imshow(depth_normalized), plt.title('normalize depths')
    plt.subplot(236), plt.imshow(depth_resized), plt.title('resize to 128x128')

    plt.show()
    break
