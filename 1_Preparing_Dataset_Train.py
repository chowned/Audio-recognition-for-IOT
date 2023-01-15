import pandas as pd
import os
import shutil
import librosa
import numpy as np
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor

# df = pd.read_csv('./dsl_data/development.csv')
# new_folder_path = './Train_Dataset_Truncated/'

df = pd.read_csv('../datasets/dsl_data/development.csv', sep=',')
new_folder_path = '../datasets/dsl_data/Train_Dataset_Truncated/'

folder_path = './dsl_data/'

###############
def process_file(file_path):
    file_path_exists = df[df["path"] == file_path].shape[0] > 0 #flag
    if file_path_exists:
        new_sr=16000
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
        y_truncated = librosa.effects.trim(y, top_db=50, frame_length=2048, hop_length=512, ref=np.max)[0]

        y_truncated = librosa.resample(y_truncated, orig_sr=sr, target_sr=new_sr)

        y_truncated = y_truncated[:int(4*new_sr)]

        target_length = 4 * new_sr
        y_truncated = librosa.util.fix_length(data=y_truncated, size=target_length) #padding

        sf.write(new_file_path, y_truncated, new_sr, 'PCM_16')

###############

if not os.path.isdir(new_folder_path):
    os.makedirs(new_folder_path) #hoping to have write permissions set

#this code will use only a thread ... and i have an AMD 5800x3d, so duck this code, infinire powaaaaaaaaaaaaaaa!
# for dirpath, dirnames, filenames in os.walk(folder_path): #all sub folders
#     # dirpath = dirpath[12:]
#     # dirpath[14] = dirpath[8]
#     # print(dirpath)
#     dirpath = dirpath.replace("\\", "/")
#     dirpath = dirpath[dirpath.index("/")+1:]

#     # print(dirpath)
#     for filename in filenames:
#         file_path = os.path.join(dirpath, filename)
#         # file_path = file_path[2:]
#         file_path = file_path.replace("\\", "/")
#         print(file_path)
#         file_path_exists = df[df["path"] == file_path].shape[0] > 0 #flag
#         # print(file_path)
#         # print(df[df["path"] == file_path].shape[0]> 0)
#         # print(df["path"]== file_path)
#         # print(df.loc[df["path"] == file_path, "Id"] > 0)
#         if file_path_exists:
#             new_sr=16000
#             print(file_path)

#             # identifier care
            
#             identifier = df.loc[df["path"] == file_path, "Id"].values[0]
#             identifier = str(int(identifier))
            
#             # label constructor

#             action = df.loc[df["path"] == file_path, "action"].values[0]
#             object = df.loc[df["path"] == file_path, "object"].values[0]
#             label  = action + object


#             new_file_path = os.path.join(new_folder_path, identifier + "_" + label + '.wav')

#             y, sr = librosa.load(file_path)
#             y_truncated = librosa.effects.trim(y, top_db=50, frame_length=2048, hop_length=512, ref=np.max)[0]

#             y_truncated = librosa.resample(y_truncated, orig_sr=sr, target_sr=new_sr)

#             y_truncated = y_truncated[:int(4*new_sr)]

#             target_length = 4 * new_sr
#             y_truncated = librosa.util.fix_length(data=y_truncated, size=target_length) #padding

#             # y_truncated, sr = librosa.load(y_truncated, sr=sr, duration=4.0)
#             # Save truncated audio
#             # librosa.output.write_wav(new_file_path, y_truncated, sr)
#             sf.write(new_file_path, y_truncated, new_sr, 'PCM_16')


#             # shutil.copy(file_path, new_file_path)

with ThreadPoolExecutor() as executor: #multi-threaded beast :)
    for dirpath, dirnames, filenames in os.walk(folder_path):
        dirpath = dirpath.replace("\\", "/")
        dirpath = dirpath[dirpath.index("/")+1:]
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            file_path = file_path.replace("\\", "/")
            executor.submit(process_file, file_path)

# print(df)
print("Execution ended")
