import os
import sounddevice as sd
import numpy as np
import time
from time import time
from time import sleep
from scipy.io.wavfile import write
import argparse as ap
import tensorflow as tf
import tensorflow_io as tfio
import uuid
import redis
import psutil
# import myConnection as mc
from datetime import datetime
import argparse as ap
import pandas as pd
import random
import noisereduce as nr
from scipy.io.wavfile import write
from scipy.io import wavfile
import paho.mqtt.client as mqtt
import sys

import uuid
mac_address = hex(uuid.getnode())

os.chdir('../') #DELETE BEFORE SEND TO PROFESSOR


try:
    os.chdir('./datasets/dsl_data/')
except:
    print("")

seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
random.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)

parser = ap.ArgumentParser()

parser.add_argument('--resolution', default=8000, type=int, help="Resolution for capturing audio")
# blocksize
#parser.add_argument('--blocksize', default=32000, type=int, help="Blocksize for captured audio, change only if you previously changed")
parser.add_argument('--downsampling_rate', default=8000, type=int, help="Resolution for capturing audio")
parser.add_argument('--device', default=0, type=int, help="Default device is 0, change for others")


parser.add_argument('--output_directory', default='./',type=str, help='Used to specify output folder')


args = parser.parse_args(['--device','31','--resolution','8000' ])
#args = parser.parse_args()

blocksize = 4 * args.resolution
LABELS = ['change languagenone', 'activatemusic', 'deactivatelights', 'increasevolume', 'decreasevolume', 'increaseheat', 'decreaseheat', 'nannan']

print(LABELS)

frame_length_in_s = 0.04#0.032*2 # /2 for resnet18
frame_step_in_s  = frame_length_in_s#frame_length_in_s

PREPROCESSING_ARGS = {
    'downsampling_rate': args.resolution,
    'frame_length_in_s': frame_length_in_s,
    'frame_step_in_s': frame_step_in_s,
}

num_mel_bins = (int) ((args.resolution - args.resolution * PREPROCESSING_ARGS['frame_length_in_s'])/(args.resolution*PREPROCESSING_ARGS['frame_step_in_s']))+1
# print(num_mel_bins)

PREPROCESSING_ARGS = {
    **PREPROCESSING_ARGS,
    'num_mel_bins': num_mel_bins,
    'lower_frequency': 20,   #40
    'upper_frequency': args.resolution/2, #4000
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

modelName = "model_24"

interpreter = tf.lite.Interpreter(model_path=f'./tflite_models/{modelName}.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

mqtt_topic = "topic/ML4IOT_Project_Polito"
client = mqtt.Client()

def connect_mqtt():
    try:
        client.connect("test.mosquitto.org",1883,60)
    except:
        print("I cannot connect to the mqtt client")
        sys.exit(0)

def get_audio_from_numpy(indata):
    indata = tf.convert_to_tensor(indata, dtype=tf.float32)
    #print("Shape of indata: ",tf.reduce_max(indata))
    indata = 2 * ((indata + 32768) / (32767 + 32768)) -1
    indata = tf.squeeze(indata)
    #print("After of indata: ",tf.reduce_max(indata))
    return indata

def get_spectrogram(indata, frame_length_in_s, frame_step_in_s):
    data = get_audio_from_numpy(indata)
    
    sampling_rate_float32 = tf.cast(args.downsampling_rate, tf.float32)
    frame_length = int(frame_length_in_s * sampling_rate_float32)
    frame_step = int(frame_step_in_s * sampling_rate_float32)

    stft = tf.signal.stft(
        data,
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=frame_length
    )
    spectrogram = tf.abs(stft)

    return spectrogram

state = False

def send_prediction_as_mqtt(predicted_label):
    # f'./{args.output_directory}/{timestamp}.wav'
    
    #print(type(predicted_label))
    #print(predicted_label.shape)
    #print(predicted_label)
    #print("predicted label:",predicted_label)
    #print("max:",predicted_label.max())
    if predicted_label.max() > 0.6:
        print(predicted_label.max()*100,"% confidence, sending label to mqtt")
        index = ( np.where(predicted_label == predicted_label.max() )  )
        index = index[0][0]
        print("The predicted label is",LABELS[index])
        client.publish(mqtt_topic, mac_address+'_'+str(index))
        
    else:
        print("Low confidence, sending noise to mqtt")
        client.publish(mqtt_topic, mac_address+'_'+'7' )
    
    #print("index",index)
    #print("label",LABELS[index])
    print()
    
def prediction_on_indata(filename):
    
    print("Prediction for",filename)
    # frame_length_in_s = 0.04
    # frame_step_in_s   = frame_length_in_s
    audio_binary = tf.io.read_file(filename)
    audio, sampling_rate = tf.audio.decode_wav(audio_binary)
    audio = tf.squeeze(audio, axis=-1) #all our audio are mono, drop extra axis
    
    #frame_length_in_s = 0.04
    #frame_step_in_s   = frame_length_in_s
    #global state
    #audio = get_audio_from_numpy(indata)
    
    frame_length = int(frame_length_in_s * args.resolution)
    frame_step = int(frame_step_in_s * args.resolution)
    stft = tf.signal.stft(
        audio,
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=frame_length
    )
    
    spectrogram = tf.abs(stft)
    
    mel_spectrogram = tf.matmul(spectrogram, linear_to_mel_weight_matrix)
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
    log_mel_spectrogram = tf.expand_dims(log_mel_spectrogram, 0)  # batch axis
    log_mel_spectrogram = tf.expand_dims(log_mel_spectrogram, -1)  # channel axis
    mfcss = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
    # print("Shape ",input_details[0])
    interpreter.set_tensor(input_details[0]['index'], mfcss) # input_details[0]['index']
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    print("change languagenone",output[0][0]*100,"%")
    print("activatemusic",output[0][1]*100,"%")
    print("deactivatelights",output[0][2]*100,"%")
    print("increasevolume",output[0][3]*100,"%")
    print("decreasevolume",output[0][4]*100,"%")
    print("increaseheat",output[0][5]*100,"%")
    print("decreaseheat",output[0][6]*100,"%")
    #print("nannan",output[0][7]*100,"%")
    
    send_prediction_as_mqtt(output[0])
    
    return state


values = sd.query_devices()
device = 0

for value in values:
    if value['name'] == 'default':
        device = value['index']

def callback2(indata, frames, callback_time, status):
    timestamp = time()
    global state
    global mac_address
    #if is_silence(indata) == 0 :
        #calculate next step of FSM!
    prediction_on_indata(indata)
    print("Elapsed time: ",time()-timestamp)

def callback(indata, frames, callback_time, status):
    global identifier
    global label
    
    # reduce noise in the audio data
    
    # write the audio data to file
    write(f'./temp.wav', args.resolution, indata)
    filesize_in_bytes = os.path.getsize(f'./temp.wav')
    filesize_in_kb = filesize_in_bytes / 1024
    print(f'Size: {filesize_in_kb:.2f}KB')
    
    
    rate,data = wavfile.read(f'./temp.wav')
    reduced_noise = nr.reduce_noise(y=data, sr=rate)
    os.remove(f'./temp.wav')
    wavfile.write(f'./temp.wav',rate,reduced_noise)
    prediction_on_indata(f'./temp.wav')

print(LABELS)

def main_test():
    with sd.InputStream(device=device, channels=1, dtype='int16', samplerate=args.resolution, blocksize=blocksize, callback=callback):
            print("") # to print a new line, improving readability in the terminal

def main():

    #print(LABELS)
    while True:
        with sd.InputStream(device=device, channels=1, dtype='int16', samplerate=args.resolution, blocksize=blocksize, callback=callback):
            print("Talk, i am hearing") # to print a new line, improving readability in the terminal

if __name__ == '__main__':
    connect_mqtt()
    main()