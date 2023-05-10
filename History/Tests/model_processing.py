import tensorflow as tf
import os
import numpy as np
import random
import tensorflow_io as tfio
import preprocessing as pr


seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
random.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)

train_ds_location = './Train_Dataset_Truncated/'
eval_ds_location  = './Test_Dataset_Truncated/'

# using Train_Dataset for both training and dataset
# Content of Test_Dataset will then be used to evaluate final accuracy
file_paths = []

for filename in os.listdir(train_ds_location):
    file_path = os.path.join(train_ds_location, filename)
    file_paths.append(file_path)
random.shuffle(file_paths)
test_percentage = 0.2
num_test_files = int(len(file_paths) * test_percentage)

train_paths = file_paths[num_test_files:] # it is shuffled, so i can do this
test_paths = file_paths[:num_test_files]

#end


train_ds = tf.data.Dataset.list_files(train_paths)
val_ds = tf.data.Dataset.list_files(eval_ds_location)
test_ds = tf.data.Dataset.list_files(test_paths)

batch_size = pr.TRAINING_ARGS['batch_size']
epochs = pr.TRAINING_ARGS['epochs']

train_ds = train_ds.map(pr.preprocess).batch(batch_size).cache()
val_ds = val_ds.map(pr.preprocess).batch(batch_size)
test_ds = test_ds.map(pr.preprocess).batch(batch_size)

for example_batch, example_labels in train_ds.take(1):
    print('Batch Shape:', example_batch.shape)
    print('Data Shape:', example_batch.shape[1:])
    print('Labels:', example_labels)
#     print('Batch Shape:', example_batch.shape)
#     print('Data Shape:', example_batch.shape[1:])
#     print('Labels:', example_labels)