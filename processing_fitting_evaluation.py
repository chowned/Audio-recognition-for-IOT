# -*- coding: utf-8 -*-

"""# Necessary install of dep and import libraries
This installs on the google colab server the necessary libraries
"""

#!pip install git > /dev/null
#!rm ./requiremen*
#!rm ./preprocessing*
#!ls
# !pip install -r ipython psutil==5.9.2 sounddevice==0.4.5 scipy==1.9.1 redis==4.3.4 tensorflow==2.10.0 tensorflow-io==0.27.0 cherrypy==18.8.0 paho-mqtt==1.6.1 > /dev/null
# !pip install -r librosa tensorflow_model_optimization pandas keras tensorflow_io > /dev/null
# !pip install tensorflow[io] > /dev/null
# !pip install tensorflow_model_optimization > /dev/null
# !pip install pydub

import pandas as pd
import os
import shutil
import librosa
import numpy as np
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor
import tensorflow as tf
import random
from tensorflow import keras
import tensorflow_model_optimization as tfmot
import argparse as ap
from pydub import AudioSegment
import sys
from tensorboard.plugins.hparams import api as hp

"""# Parameters setup
This code begins as a simple python script. Then we moved to jupyter. Jupyter doesn't support the argument parser, but it is still good to have in case in the future we want to run it as a script.
"""
new_sr=8000
folder_path = './audio'
LABELS = []
num_units = 512

os.chdir('./datasets/dsl_data/')

parser = ap.ArgumentParser()

parser.add_argument('--batch_size', default=32, type=int, help="Choosing batch size default is 32")
parser.add_argument('--initial_learning_rate', default=0.03, type=float, help="Choosing initial_learning_rate")
parser.add_argument('--end_learning_rate', default=0.001, type=float, help="Choosing end_learning_rate")
parser.add_argument('--epochs', default=50, type=int, help="Choosing epochs")
parser.add_argument('--test_percentage', default=0.20, type=float, help="Choosing test_percentage")
parser.add_argument('--pruning_initial_step', default=0.2, type=float, help="Choosing pruning_initial_step")
parser.add_argument('--initial_sparsity', default=0.40, type=float, help="Choosing initial_sparsity")
parser.add_argument('--alpha', default=1, type=float, help="Choosing alpha")

parser.add_argument('--eval_percentage', default=0.0, type=float, help="Choosing eval_percentage")
#,'--eval_percentage','0.0'

"""Parser arguments"""

args = parser.parse_args(['--epochs','200','--alpha','0.5','--batch_size','32','--pruning_initial_step','0.9','--initial_learning_rate','0.01','--end_learning_rate','0.005'])
# args = parser.parse_args()

"""# Preprocessing HP
These HP are responsible for the mel bins. frame_length_in_s is one of the most important
"""

frame_length_in_s = 0.032*2 # /2 for resnet18
frame_step_in_s  = frame_length_in_s

PREPROCESSING_ARGS = {
    'downsampling_rate': new_sr,
    'frame_length_in_s': frame_length_in_s,
    'frame_step_in_s': frame_step_in_s,
}

num_mel_bins = (int) ((new_sr - new_sr * PREPROCESSING_ARGS['frame_length_in_s'])/(new_sr*PREPROCESSING_ARGS['frame_step_in_s']))+1
# print(num_mel_bins)

PREPROCESSING_ARGS = {
    **PREPROCESSING_ARGS,
    'num_mel_bins': num_mel_bins,
    'lower_frequency': 40,
    'upper_frequency': 4000,
}

downsampling_rate = PREPROCESSING_ARGS['downsampling_rate']
sampling_rate_int64 = tf.cast(downsampling_rate, tf.int64)
frame_length = int(downsampling_rate * PREPROCESSING_ARGS['frame_length_in_s'])
#print("Frame_length: {}".format(frame_length))
frame_step = int(downsampling_rate * PREPROCESSING_ARGS['frame_step_in_s'])
#print("Frame_length: {}".format(frame_step))
num_spectrogram_bins = frame_length // 2 + 1
num_mel_bins = PREPROCESSING_ARGS['num_mel_bins']
lower_frequency = PREPROCESSING_ARGS['lower_frequency']
upper_frequency = PREPROCESSING_ARGS['upper_frequency']

linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
    num_mel_bins=num_mel_bins,
    num_spectrogram_bins=num_spectrogram_bins,
    sample_rate=downsampling_rate,
    lower_edge_hertz=lower_frequency,
    upper_edge_hertz=upper_frequency
)

def preprocess(filename):
    audio_binary = tf.io.read_file(filename)

    path_parts = tf.strings.split(filename, '_')
    path_end = path_parts[-1]
    file_parts = tf.strings.split(path_end, '.')
    true_label = file_parts[0]
    label_id = tf.argmax(true_label == LABELS)
    audio, sampling_rate = tf.audio.decode_wav(audio_binary)
    audio = tf.squeeze(audio, axis=-1) #all our audio are mono, drop extra axis
    audio_padded = audio
    stft = tf.signal.stft(
        audio,
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=frame_length
    )
    spectrogram = tf.abs(stft)
    mel_spectrogram = tf.matmul(spectrogram, linear_to_mel_weight_matrix)
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
    log_mel_spectrogram = tf.expand_dims(log_mel_spectrogram, -1)  # channel axis
    mfcss = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)

    return mfcss, label_id

"""# Code for finding optimal length of audio files
As we will re-use the code in the future, this part of the code will automatically decide where to "cut" the length of the audio dataset. 

1.   Use function scan folder to scan the dataset audio folder
2.   This function returns the audio files duration -> (1s , 400 files) (2s, 300 files)
3.   Use function "find duration" to find optimal length. In particular, we want 90% of audio files to be included in the duration that we will get.
"""
class MyThresholdCallback(tf.keras.callbacks.Callback):
    def __init__(self, threshold):
        super(MyThresholdCallback, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None): 
        val_acc = logs["val_sparse_categorical_accuracy"]
        if val_acc >= self.threshold:
            self.model.stop_training = True


def scan_folder(folder):
  duration_count = {}
  for root, dirs, files in os.walk(folder):
    for file in files:
      if file.endswith(".wav"):
        file_path = os.path.join(root, file)
        audio = AudioSegment.from_wav(file_path)
        duration = len(audio)
        if duration in duration_count:
          duration_count[duration] += 1
        else:
            duration_count[duration] = 1
  return duration_count

def create_dataframe(duration_count):
  data = {"Duration of audio file": list(duration_count.keys()), 
            "Number of audio files with that duration": list(duration_count.values())}
  df = pd.DataFrame(data)
  df = df.sort_values(by='Number of audio files with that duration', ascending=False)
  return df


# find the percentage. The duration returned in second is the size that include 1-percentage inside

def find_duration(folder_path, percentage_files=0.93):
  duration_count = {}
  for root, dirs, files in os.walk(folder_path):
    for file in files:
      if file.endswith(".wav"):
        file_path = os.path.join(root, file)
        #print(file_path)
        audio = AudioSegment.from_wav(file_path)
        duration = len(audio) / 1000 #convert from ms to sec
        if duration in duration_count:
          duration_count[duration] += 1
        else:
          duration_count[duration] = 1
    total_files = sum(duration_count.values())
    target_files = total_files * percentage_files
    current_count = 0
    for duration, count in sorted(duration_count.items()):
      current_count += count
      if current_count >= target_files:
        duration = round(duration)
        print(f"Duration of audio that makes {percentage_files*100}% of the files have that duration is: {duration} seconds")
        return duration
    
# process_file better to be implemented here with a boolean value that checks if i am processing train_dataset or eval file
def process_file(file_path, flag):
    file_path_exists = df[df["path"] == file_path].shape[0] > 0 #flag
    if file_path_exists:
        # new_sr=16000
        # identifier care
        identifier = df.loc[df["path"] == file_path, "Id"].values[0]
        identifier = str(int(identifier))
        # label constructor
        label = ""
        if flag == 1:
            label  += "_"
            action  = df.loc[df["path"] == file_path, "action"].values[0]
            object  = df.loc[df["path"] == file_path, "object"].values[0]
            label  += action + object
        # If no label available, code will just go on
        new_file_path = os.path.join(new_folder_path, identifier + label + '.wav')
        #print(new_file_path)
        y, sr = librosa.load('../'+file_path)
        #print('../'+file_path)
        y_truncated = librosa.effects.trim(y, top_db=50, frame_length=2048, hop_length=512, ref=np.max)[0]
        y_truncated = librosa.resample(y_truncated, orig_sr=sr, target_sr=new_sr)
        y_truncated = y_truncated[:int(length_calculated*new_sr)] #if longer
        target_length = length_calculated * new_sr
        y_truncated = librosa.util.fix_length(data=y_truncated, size=target_length) # padding, if shorter
        sf.write(new_file_path, y_truncated, new_sr, 'PCM_16')

duration = find_duration(folder_path)
length_calculated = duration



""" # Preprocessing for Train dataset files
 This part of the code will cut the original dataset for the desired length that we found before.


1.   First we create a new audio file, the name will be "identifier + label + '.wav'" -> "0_increasevolume.wav"
2.   y_truncated = librosa.effects.trim(y, top_db=50, frame_length=2048, hop_length=512, ref=np.max)[0]
Then we trim the audio, we delete the parts that have silence
3.   y_truncated = librosa.resample(y_truncated, orig_sr=sr, target_sr=new_sr)
We change sampling rate to 16000, that is the one we want to use
4.   y_truncated = y_truncated[:int(length_calculated*new_sr)] #if longer

Then we cut in case one audio file was longer than 4s and point 3 didn't make it shorter. At the end, we absolutely want audio files that are 4s.
"""


df = pd.read_csv('./development.csv', sep=',')
new_folder_path = './Train_Dataset_Truncated/'
folder_path = '../dsl_data/audio/'

if not os.path.isdir(new_folder_path):
  os.makedirs(new_folder_path) # hoping to have write permissions set
if not os.listdir(new_folder_path):
  print("Creating dataset files")
  with ThreadPoolExecutor() as executor: # who is your single threaddy?
    for dirpath, dirnames, filenames in os.walk(folder_path):
      dirpath = dirpath.replace("\\", "/")
      dirpath = dirpath[dirpath.index("/")+1:] 
      #dirpath = dirpath[dirpath.index("/")+1:] 
      #print(dirpath)
      #dirpath = dirpath[dirpath.index("/")+1:]
      #print(dirpath)
      for filename in filenames:
        file_path = os.path.join(dirpath, filename)
        file_path = file_path.replace("\\", "/")
        executor.submit(process_file, file_path, 1)
# print(df)
#print("Execution ended")
#rint(file_path)

""" # Preprocessing for Evaluation dataset files
 The same as before but for the evaluation dataset to send
"""

df = pd.read_csv('./evaluation.csv', sep=',')
new_folder_path = './Test_Dataset_Truncated/'
folder_path = '../dsl_data/audio/'

if not os.path.isdir(new_folder_path):
    os.makedirs(new_folder_path)
if not os.listdir(new_folder_path):
    print("Creating evaluation files")
    with ThreadPoolExecutor() as executor:
        for dirpath, dirnames, filenames in os.walk(folder_path):
            dirpath = dirpath.replace("\\", "/")
            dirpath = dirpath[dirpath.index("/")+1:]
            dirpath = dirpath
            #dirpath = dirpath[dirpath.index("/")+1:]
            # print(dirpath)
            #dirpath = dirpath[dirpath.index("/")+1:]
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                file_path = file_path.replace("\\", "/")
                executor.submit(process_file, file_path, 0)

# print("Execution ended")



"""# Auto - updating labels
This part of the code is responsible for getting the labels. The labels are not decided a priori, they will be "calculated" from the development.csv file and stored in a list.
"""

df = pd.read_csv('./development.csv', sep=',')
df['labels'] = df['action'].astype(str) + df['object'].astype(str)
distinct_values = df['labels'].unique()

LABELS = distinct_values.tolist()
# print("Labels to predict: ",LABELS)




"""## This part of the code exist to manage all the folders
## Please be careful, if the directories tree is not respected, the code will not work properly
"""

seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
random.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)

"""creates the folder for tensorboard. It calculates how many times we run before, this is useful to give a new name for tensorboard. If it is my first time i run the code, tb_run = 0. The second time i run, tb_run = 1."""

# Useful to save tensorboard data
log_dir_tensorboard = './tensorboard_data/'
if not os.path.isdir(log_dir_tensorboard):
    os.makedirs(log_dir_tensorboard)
runs = [int(d.split('_')[1]) for d in os.listdir(log_dir_tensorboard) if 'run_' in d]
tb_run = max(runs) + 1 if runs else 0

"""These folder are the dataset folder that we must use and the folders to save models and checkpoint. If they do not exist, i create them"""

# Folder creation
train_ds_location      = './Train_Dataset_Truncated/'
log_dir_model          = './models/'
#run_{}_
model_name             = 'frame_l_{}_epochs_{}_batch_size_{}_pruning_initial_step_{}_initial_learning_rate_{}_end_learning_rate_{}_test_percentage_{}_pruning_initial_step_{}_initial_sparsity_{}_alpha_{}'.format(frame_length_in_s,args.epochs,args.batch_size,args.pruning_initial_step,args.initial_learning_rate,args.end_learning_rate,args.test_percentage,args.pruning_initial_step,args.initial_sparsity,args.alpha)
checkpoint_path        = './checkpoints/' + model_name
#check_point_file_name  = checkpoint_path+'.ckpt'

# If folders to not exist -> create them
# This code will not check for the dataset folders, the code above must be executed
if not os.path.isdir(log_dir_model):
    os.makedirs(log_dir_model)
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)

""" # Obtaining Test data from train data, using shuffle and avoiding retaking same data on different runs

 As I only have a dataset, and i want to have "train dataset", "test dataset" and "eval dataset", what i do is creating the list "file_paths" that contains all the files ("0_decreasevolume.wav"). Then what i do is random.shuffle(file_paths), that is I randomly mix them. Then, i take the percentages that i want (given by the parser).
"""

file_paths = []

for filename in os.listdir(train_ds_location):
    file_path = os.path.join(train_ds_location, filename)
    file_paths.append(file_path)
random.shuffle(file_paths)
num_test_files = int(len(file_paths) * args.test_percentage)
num_eval_files = int(len(file_paths) * args.eval_percentage)
#not using eval dataset


# num_eval_files = num_eval_files

# it is shuffled, so i can do this
test_paths     = file_paths[:num_test_files]                 # from 0 to num_test_files
train_paths    = file_paths[num_test_files:]
#train_paths    = file_paths[num_test_files:-num_eval_files]  # from num_test_files to end-num_eval_files
eval_paths     = file_paths[-num_eval_files:]                # until the end

# print(len(train_paths))
# print(len(test_paths))
# print(len(eval_paths))

"""# Preprocessing data and model creation"""

train_ds       = tf.data.Dataset.list_files(train_paths)
val_ds         = tf.data.Dataset.list_files(eval_paths)
test_ds        = tf.data.Dataset.list_files(test_paths)

train_ds       = train_ds.map(preprocess).batch(args.batch_size).cache()
val_ds         = val_ds.map(preprocess).batch(args.batch_size)
test_ds        = test_ds.map(preprocess).batch(args.batch_size)

for example_batch, example_labels in train_ds.take(1):
  print('Batch Shape:', example_batch.shape)
  print('Data Shape:', example_batch.shape[1:])
  print('Labels:', example_labels)

# prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
# begin_step          = int(len(train_ds) * args.epochs * args.pruning_initial_step)
# end_step            = int(len(train_ds) * args.epochs)
# pruning_params      = {
#     'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
#         initial_sparsity=args.initial_sparsity,
#         final_sparsity=final_sparsity,
#         begin_step=begin_step,
#         end_step=end_step
#     )
# }
# custom_objects      = {'PruneLowMagnitude': prune_low_magnitude}

# model_name          = 'model_'+str(args.batch_size)+'_'+str(args.alpha)
# model_name += '.h5'

hparams = {
'num_units' : num_units,
'alpha_rate': args.alpha,
'frame l'   : frame_length_in_s,
'epochs'    : args.epochs,
'batch_size': args.batch_size,
}

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=example_batch.shape[1:]),
    tf.keras.layers.Conv2D(filters=int(num_units * args.alpha), kernel_size=[3, 3], strides=[2, 2],
        use_bias=False, padding='valid'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Conv2D(filters=int(num_units * args.alpha), kernel_size=[3, 3], strides=[1, 1],
            use_bias=False, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Conv2D(filters=int(num_units * args.alpha), kernel_size=[3, 3], strides=[1, 1],
        use_bias=False, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(units=len(LABELS)),
    tf.keras.layers.Softmax()
    ])

#example_batch = example_batch.reshape(-1, example_batch.shape[1:])
#example_batch = np.concatenate([example_batch, example_batch, example_batch], axis=-1)
#model = tf.keras.applications.resnet50.ResNet50(weights='imagenet', include_top=False, input_shape=(example_batch.shape[1],example_batch.shape[2],example_batch.shape[3]))


# model_for_pruning = prune_low_magnitude(model, **pruning_params)

# print('Batch Shape:', example_batch.shape)
# print('Data Shape:', example_batch.shape[1])
# print('Data Shape:', example_batch.shape[2])
# print('Data Shape:', example_batch.shape[3])
# print('Labels:', example_labels)

# this model uses Transfer Learning... I mean, we transferred a model developed for another course to this course
# print(example_batch.shape[1:])
# model_for_pruning.summary()

"""# Model fitting"""

loss = tf.losses.SparseCategoricalCrossentropy(from_logits=False)

linear_decay = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=args.initial_learning_rate,
    end_learning_rate=args.end_learning_rate,
    decay_steps=len(train_ds) * args.epochs,
)
optimizer = tf.optimizers.Adam(learning_rate=linear_decay)
metrics = [tf.metrics.SparseCategoricalAccuracy()]
tensorboard_model_saved = f"run_{tb_run}"


my_callback = MyThresholdCallback(threshold=0.90)

callbacks = [ tf.keras.callbacks.ModelCheckpoint(filepath=log_dir_tensorboard+model_name+'.ckpt',save_weights_only=True,verbose=1),
             tfmot.sparsity.keras.UpdatePruningStep(), 
             keras.callbacks.TensorBoard(log_dir=log_dir_tensorboard+tensorboard_model_saved, histogram_freq=1) , 
             hp.KerasCallback(log_dir_tensorboard+tensorboard_model_saved, hparams),# val_accuracy
             #tf.keras.callbacks.EarlyStopping(monitor='sparse_categorical_accuracy', mode='max', patience=10, min_delta=2.0, restore_best_weights=True, verbose=1, baseline=0.985),
             my_callback,
             ]


model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

if os.path.exists(log_dir_tensorboard+model_name+'.ckpt'):
    print("Checkpoint found, loading...")
    model.load_weights(log_dir_tensorboard+model_name+'.ckpt')
    with open(log_dir_tensorboard+model_name+"epochs.txt", "r") as file:
        contents = file.read()
        previous_epoch_run = int(contents)
        previous_epoch_run = previous_epoch_run
    print("Restoring from epoch : {}".format(previous_epoch_run))
else:
    print("No previous check_point found.")
    previous_epoch_run = 0

#validation data is test_ds validation_data=val_ds,
    
history = model.fit(train_ds, validation_data=test_ds, epochs=args.epochs, callbacks=callbacks,verbose=1,initial_epoch=previous_epoch_run) #it was valds

with open(log_dir_tensorboard+model_name+"epochs.txt", "w") as file:
    file.write(str(args.epochs))

"""# Evaluation on created eval Dataset"""

# Evaluation on created eval Dataset

test_loss, test_accuracy = model.evaluate(test_ds)

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

with open(log_dir_model+model_name+".txt", "w") as file:
    file.write(model_name)
    file.write("\n")
    file.write("Frame length =  {}".format(frame_length_in_s))
    file.write("\n")
    file.write("Execution lasted: " + str(args.epochs))
    file.write("\n")
    file.write(f'\nTraining Loss: {training_loss:.4f}')
    file.write(f'\nTraining Accuracy: {training_accuracy*100.:.2f}%')
    file.write("\n")
    file.write(f'\nValidation Loss: {val_loss:.4f}')
    file.write(f'\nValidation Accuracy: {val_accuracy*100.:.2f}%')
    file.write("\n")
    file.write(f'\nTest Loss: {test_loss:.4f}')
    file.write(f'\nTest Accuracy: {test_accuracy*100.:.2f}%')
    
saved_model_dir = f'./saved_models/last_model_used'
if not os.path.exists(saved_model_dir):
    os.makedirs(saved_model_dir)
model.save(saved_model_dir)

print(log_dir_model+model_name)
print("run: ",tb_run)
with open(log_dir_model+model_name+".txt", "r") as file:
        contents = file.read()
        print(contents)

"""# Evaluation of submitted data"""

from glob import glob
filenames = glob('./Test_Dataset_Truncated/*')

with open(f"Evaluation_Dataset_Result_{tb_run}.csv", "w") as file:
    file.write("Id,Predicted")
    #file.write("\n") 
    file.write("")
    for filename in filenames:
        identifier = filename.replace("\\", "/").split('/')[-1].split('.')[0]
        #filename = filename.split('/')[-1].split('.')[0]
        #print(identifier)
        audio_binary = tf.io.read_file(filename)
        audio, sampling_rate = tf.audio.decode_wav(audio_binary)
        audio = tf.squeeze(audio, axis=-1) #all our audio are mono, drop extra axis
        audio_padded = audio
        stft = tf.signal.stft(
            audio,
            frame_length=frame_length,
            frame_step=frame_step,
            fft_length=frame_length
        )
        spectrogram = tf.abs(stft)
        mel_spectrogram = tf.matmul(spectrogram, linear_to_mel_weight_matrix)
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
        log_mel_spectrogram = tf.expand_dims(log_mel_spectrogram, -1)  # channel axis
        mfcss = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
        
        mfcss = tf.expand_dims(mfcss, 0)
        prediction = model.predict(mfcss)
        
        prediction = np.argmax(prediction[0])
        prediction = LABELS[prediction]
        
        #print(prediction)
        file.write("\n{},{}".format(identifier,prediction))