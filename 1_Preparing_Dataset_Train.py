import pandas as pd
import os
import shutil
import librosa
import numpy as np
import soundfile as sf

df = pd.read_csv('./dsl_data/development.csv')
new_folder_path = './Train_Dataset_Truncated/'

folder_path = './dsl_data/'

if not os.path.isdir(new_folder_path):
    os.makedirs(new_folder_path) #hoping to have write permissions set

for dirpath, dirnames, filenames in os.walk(folder_path): #all sub folders
    for filename in filenames:
        file_path = os.path.join(dirpath, filename)
        file_path = file_path[2:]
        
        file_path_exists = df[df["path"] == file_path].shape[0] > 0 #flag
        # print(file_path)
        if file_path_exists:
            
            # print(file_path)

            # identifier care
            
            identifier = df.loc[df["path"] == file_path, "Id"].values[0]
            identifier = str(int(identifier))
            
            # label constructor

            action = df.loc[df["path"] == file_path, "action"].values[0]
            object = df.loc[df["path"] == file_path, "object"].values[0]
            label  = action + object


            new_file_path = os.path.join(new_folder_path, identifier + "_" + label + '.wav')

            y, sr = librosa.load(file_path)
            y_truncated = librosa.effects.trim(y, top_db=30, frame_length=2048, hop_length=512, ref=np.max)[0]

            # Save truncated audio
            # librosa.output.write_wav(new_file_path, y_truncated, sr)
            sf.write(new_file_path, y_truncated, sr, 'PCM_24')


            # shutil.copy(file_path, new_file_path)

print("Execution ended")
