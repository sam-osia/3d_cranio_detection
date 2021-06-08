import sys
sys.path.insert(0, '..')
from utils.utils import *

import h5py
import cv2



# filename = '/home/saman/Desktop/cts-mount/final_data/3D/3D_data.h5.256'
filename = '/hpf/largeprojects/ccm/devin/cts/final_data/3D/3D_data.h5.256'
dataset = h5py.File(filename, 'r')

data, data_si, target = dataset.get('data_im')[()], dataset.get('data_si')[()], dataset.get('target')[()].astype(int)

data = np.asarray(data)

data, data_si, target = remove_images(data, data_si, target)
data, data_si, target = drop_post_op_patients(data, data_si, target)
data, data_si, target = drop_patients_above_age(data, data_si, target, age=1)
data_si, target = fix_incorrect_labels(data_si, target)

print(data.shape)
print(data_si.shape)
print(target.shape)

# for i in range(10):
#     sample_data = data[i]
#
#     axis_0 = np.sum(sample_data, axis=0)
#     axis_1 = np.sum(sample_data, axis=1)
#     axis_2 = np.sum(sample_data, axis=2)
#
#     print(axis_0.shape)
#     print(axis_1.shape)
#     print(axis_2.shape)
#
#     parent_dir = '/hpf/largeprojects/ccm/devin/cts/sam/3d_cranio_detection/data/temp'
#
#     np.save(os.path.join(parent_dir, f'{i}_axis_0.npy'), axis_0)
#     np.save(os.path.join(parent_dir, f'{i}_axis_1.npy'), axis_1)
#     np.save(os.path.join(parent_dir, f'{i}_axis_2.npy'), axis_2)
#
#     cv2.imwrite(os.path.join(parent_dir, f'{i}_axis_0.png'), np.reshape(axis_0, (256, 256, 1)))
#     cv2.imwrite(os.path.join(parent_dir, f'{i}_axis_1.png'), np.reshape(axis_1, (256, 256, 1)))
#     cv2.imwrite(os.path.join(parent_dir, f'{i}_axis_2.png'), np.reshape(axis_2, (256, 256, 1)))
