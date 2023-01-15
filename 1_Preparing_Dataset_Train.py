import pandas as pd
import os
import shutil
import librosa
import numpy as np
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor

# df = pd.read_csv('./dsl_data/development.csv')
# new_folder_path = './Train_Dataset_Truncated/'

df = pd.read_csv('../../datasets/dsl_data/development.csv', sep=',')
new_folder_path = '../../datasets/dsl_data/Train_Dataset_Truncated/'

folder_path = '../../datasets/dsl_data/'

###############
def process_file(file_path):
    file_path_exists = df[df["path"] == file_path].shape[0] > 0 #flag
    # print(file_path)
    # print(file_path_exists)
    if file_path_exists:
        new_sr=16000
        # print(file_path)

        # identifier care
        identifier = df.loc[df["path"] == file_path, "Id"].values[0]
        identifier = str(int(identifier))
        # print("\n "+identifier)

        # label constructor

        action = df.loc[df["path"] == file_path, "action"].values[0]
        object = df.loc[df["path"] == file_path, "object"].values[0]
        label  = action + object

        new_file_path = os.path.join(new_folder_path, identifier + "_" + label + '.wav')

        # print(file_path)
        y, sr = librosa.load('../../datasets/'+file_path)
        # print("here")
        # print(y)
        y_truncated = librosa.effects.trim(y, top_db=50, frame_length=2048, hop_length=512, ref=np.max)[0]
        # print(y_truncated)
        y_truncated = librosa.resample(y_truncated, orig_sr=sr, target_sr=new_sr)

        y_truncated = y_truncated[:int(4*new_sr)] #if longer

        target_length = 4 * new_sr
        y_truncated = librosa.util.fix_length(data=y_truncated, size=target_length) #padding, if shorter

        sf.write(new_file_path, y_truncated, new_sr, 'PCM_16')

if not os.path.isdir(new_folder_path):
    os.makedirs(new_folder_path) #hoping to have write permissions set

with ThreadPoolExecutor(max_workers=4) as executor: #multi-threaded beast :)
    for dirpath, dirnames, filenames in os.walk(folder_path):
        dirpath = dirpath.replace("\\", "/")
        dirpath = dirpath[dirpath.index("/")+1:]
        dirpath = dirpath[dirpath.index("/")+1:]
        dirpath = dirpath[dirpath.index("/")+1:]
        # print(dirpath)
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            file_path = file_path.replace("\\", "/")
            # print(file_path)
            # process_file(file_path)
            executor.submit(process_file, file_path)

# print(df)
print("Execution ended")
