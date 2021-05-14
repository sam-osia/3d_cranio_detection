import sys
import time
from pathlib import Path
import numpy as np
import os
from math import sin, cos

from preprocess import binvox_rw


# region General Utility Functions
def set_path(user='auto'):
    if user == 'auto':
        user = detect_user()

    print(f'Setting path for {user}...')

    if user == 'saman':     # saman's personal computer
        recursive_unix_dir_backtrack('3d_cranio_detection')
    elif user == 'samosia':   # vector cluster
        recursive_unix_dir_backtrack('3d_cranio_detection')
    else:
        raise Exception('unable to recognize user')

    print(os.getcwd())


def recursive_unix_dir_backtrack(desired_dir):
    dir_name = os.getcwd().split('/')[-1]
    if dir_name != desired_dir:
        os.chdir('..')
        recursive_unix_dir_backtrack(desired_dir)


def detect_user():
    users = ['saman', 'samosia']
    exec_path = sys.executable.lower()
    user = None
    for u in users:
        if u in exec_path:
            user = u

    if user is None:
        raise Exception('unable to detect user')
    return user


def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def overrides(interface_class):
    def overrider(method):
        assert(method.__name__ in dir(interface_class))
        return method
    return overrider


def load_binvox(path):
    with open(path, 'rb') as f:
        model = binvox_rw.read_as_3d_array(f)

    return model.data


def load_numpy(path):
    return np.load(path)


# endregion

# region 2D mapping functions
def get_density(m, axis):
    return np.sum(m.data, axis=axis)


def get_contour(m, axis):
    density = get_density(m, axis=axis)
    density[density > 0] = 255
    return density


def get_depths(m, axis, flip=True):
    m_adjusted = m.copy()
    if flip:
        if axis == 0:
            m_adjusted = m[::-1, :, :]
        elif axis == 1:
            m_adjusted = m[:, ::-1, :]
        elif axis == 2:
            m_adjusted = m[:, :, ::-1]

    axes = [0, 1, 2]  # all the axes
    axes.remove(axis)  # remove the axis we're looking along
    points = np.where(m_adjusted == 1)  # get the 3D coordinate of all points in the point cloud

    # coords is the 2D array of all the points
    # depth is the corresponding depth of the point at each coord in coords
    coords = np.stack((points[axes[0]], points[axes[1]]), axis=1)
    depth = points[axis]

    # get the unique coordinates because we're only interested in the closest one
    coords_unique_ind = np.unique(coords, axis=0, return_index=True)[1]

    # apply the unique indices to the coordinates and depths list
    unique_coords = coords[coords_unique_ind]
    unique_depths = depth[coords_unique_ind]

    # map out the depths in a 2D array
    depth_map = np.zeros((m_adjusted.shape[axes[0]], m_adjusted.shape[axes[1]]), dtype='uint8')
    depth_map[unique_coords.T.tolist()] = unique_depths

    # flip the numbers to show closer as higher number
    # TODO: need to apply the line below on a mask that doesn't contain the contour of the image, otherwise points
    #  within the person with a value of 0 don't get flipped
    # depth_map[depth_map != 0] = 255 - depth_map[depth_map != 0]
    return depth_map

# endregion


# region Numpy Utility Functions
def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def multi_dim_padding(a: np.array, desired_shape):
    def pad_with(vector, pad_width, iaxis, kwargs):
        pad_value = kwargs.get('padder', 0)
        vector[:pad_width[0]] = pad_value
        vector[-pad_width[1]:] = pad_value

    if len(a.shape) != len(desired_shape):
        raise Exception('Please make sure the array and desired shape are of the same rank!')

    for i in range(len(desired_shape)):
        if a.shape[i] > desired_shape[i]:
            raise Exception(f'Array shape larger than desired shape on axis {i}')

    padding_widths = []
    for i in range(len(desired_shape)):
        ax_before = int((desired_shape[i] - a.shape[i]) / 2)
        ax_after = int(desired_shape[i] - a.shape[i] - ax_before)
        padding_widths.append((ax_before, ax_after))

    print(padding_widths)
    return np.pad(a, padding_widths, pad_with)

# endregion


# region Model utility functions
def get_log_dir(parent_dir, model_name):
    run_id = time.strftime(f'{model_name}_%Y_%m_%d-%H_%M_%S')
    return os.path.join(parent_dir, run_id)


def get_save_dir(parent_dir, run_name):
    return os.path.join(parent_dir, run_name + '.h5')

# endregion


if __name__ == '__main__':
    a = np.arange(1, 9)
    a = a.reshape((2, 2, 2))

    print(multi_dim_padding(a, (4, 4, 4)))
