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

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=example_batch.shape[1:]),
    tf.keras.layers.Conv2D(filters=int(128 * pr.alpha), kernel_size=[3, 3], strides=[2, 2],
        use_bias=False, padding='valid'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Conv2D(filters=int(128 * pr.alpha), kernel_size=[3, 3], strides=[1, 1],
            use_bias=False, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Conv2D(filters=int(128 * pr.alpha), kernel_size=[3, 3], strides=[1, 1],
        use_bias=False, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(units=len(pr.LABELS)),
    tf.keras.layers.Softmax()
])

import tensorflow_model_optimization as tfmot

prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

begin_step = int(len(train_ds) * epochs * 0.2)
end_step = int(len(train_ds) * epochs)

pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.40,
        final_sparsity=pr.final_sparsity,
        begin_step=begin_step,
        end_step=end_step
    )
}

model_for_pruning = prune_low_magnitude(model, **pruning_params)

#model_for_pruning.summary()

loss = tf.losses.SparseCategoricalCrossentropy(from_logits=False)
initial_learning_rate = pr.TRAINING_ARGS['initial_learning_rate']
end_learning_rate = pr.TRAINING_ARGS['end_learning_rate']

linear_decay = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=initial_learning_rate,
    end_learning_rate=end_learning_rate,
    decay_steps=len(train_ds) * epochs,
)
optimizer = tf.optimizers.Adam(learning_rate=linear_decay)
metrics = [tf.metrics.SparseCategoricalAccuracy()]
callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
model_for_pruning.compile(loss=loss, optimizer=optimizer, metrics=metrics)

history = model_for_pruning.fit(train_ds, epochs=epochs, validation_data=test_ds,callbacks=callbacks) #it was valds

test_loss, test_accuracy = model_for_pruning.evaluate(test_ds)

training_loss = history.history['loss'][-1]
training_accuracy = history.history['sparse_categorical_accuracy'][-1]
val_loss = history.history['val_loss'][-1]
val_accuracy = history.history['val_sparse_categorical_accuracy'][-1]

print(f'Training Loss: {training_loss:.4f}')
print(f'Training Accuracy: {training_accuracy*100.:.2f}%')
print()
print(f'Validation Loss: {val_loss:.4f}')
print(f'Validation Accuracy: {val_accuracy*100.:.2f}%')
print()
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy*100.:.2f}%')