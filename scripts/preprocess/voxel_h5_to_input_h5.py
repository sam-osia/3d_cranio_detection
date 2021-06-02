import h5py
import sys

sys.path.insert(0, '..')
from utils.utils import *

# filename = '/home/saman/Desktop/cts-mount/final_data/3D/3D_data.h5.256'
filename = '/hpf/largeprojects/ccm/devin/cts/final_data/3D/3D_data.h5.256'
dataset = h5py.File(filename, 'r')

target = dataset.get('data_im')[()][:1]

print(type(target))
print(target.shape)
