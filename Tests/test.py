import os

folder_path = './dsl_data'

objects = os.listdir(folder_path)

# for obj in objects:
#     print(obj)

# for obj in objects:
#     if os.path.isfile(os.path.join(folder_path, obj)):
#         print(obj)

# for dirpath, dirnames, filenames in os.walk(folder_path):
#     for filename in filenames:
#         print(os.path.join(dirpath, filename))

import os
import wave

directory = './Train_Dataset_Truncated/'

max_duration = 0
max_filename = ''

for filename in os.listdir(directory):
    if filename.endswith('.wav'):
        wav_path = os.path.join(directory, filename)
        with wave.open(wav_path, 'r') as wav:
            duration = wav.getparams()[3] / wav.getparams()[2]
            if duration > max_duration:
                max_duration = duration
                max_filename = wav_path+filename

print("Maximum duration:", max_duration)
print("Filename:", max_filename)

# 4699 also with sampling rate 22050