import tensorflow as tf
print(tf.__version__)
print(tf.test.is_built_with_cuda())
print(len(tf.config.list_physical_devices('GPU')))
print(tf.config.list_physical_devices('GPU'))
