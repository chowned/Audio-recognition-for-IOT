import pandas as pd
import os
import shutil

df = pd.read_csv('evaluation.csv')
new_folder_path = './Test_Dataset/'

folder_path = './dsl_data/'

for dirpath, dirnames, filenames in os.walk(folder_path): #all sub folders
    for filename in filenames:
        file_path = os.path.join(dirpath, filename)
        file_path = file_path[2:]
        
        file_path_exists = df[df["path"] == file_path].shape[0] > 0 #flag
        # print(file_path)
        if file_path_exists:
            
            print(file_path)
            identifier = df.loc[df["path"] == file_path, "Id"].values[0]
            identifier = str(int(identifier))
            new_file_path = os.path.join(new_folder_path, identifier + '.wav')

            shutil.copy(file_path, new_file_path)

print("Execution ended")
