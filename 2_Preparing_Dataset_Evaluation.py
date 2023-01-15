import pandas as pd
import os
import shutil

df = pd.read_csv('../datasets/dsl_data/evaluation.csv', sep=',') # just to be sure i set the sep parameter
new_folder_path = '../datasets/dsl_data/Test_Dataset_Truncated/'

folder_path = '../datasets/dsl_data/'

if not os.path.exists(new_folder_path):
    os.makedirs(new_folder_path)

for dirpath, dirnames, filenames in os.walk(folder_path): #all sub folders
    dirpath = dirpath[12:] #remove bebinning of path
    # dirpath[14] = dirpath[8]
    dirpath = dirpath.replace("\\", "/")
    for filename in filenames:
        file_path = os.path.join(dirpath, filename)
        # file_path = file_path[2:]
        file_path = file_path.replace("\\", "/")
        print(file_path)
        
        file_path_exists = df[df["path"] == file_path].shape[0] > 0 #flag
        # print(file_path)
        if file_path_exists:
            
            print(file_path)
            identifier = df.loc[df["path"] == file_path, "Id"].values[0]
            identifier = str(int(identifier))
            new_file_path = os.path.join(new_folder_path, identifier + '.wav')

            shutil.copy(file_path, new_file_path)

print("Execution ended")
