import pandas as pd
import os
import shutil

df = pd.read_csv('../datasets/dsl_data/development.csv', sep=',')
new_folder_path = '../datasets/dsl_data/Train_Dataset_Truncated/'

folder_path = '../datasets/dsl_data/'

if not os.path.exists(new_folder_path):
    os.makedirs(new_folder_path)

# print(df)

for dirpath, dirnames, filenames in os.walk(folder_path): #all sub folders
    dirpath = dirpath[12:]
    # dirpath[14] = dirpath[8]
    dirpath = dirpath.replace("\\", "/")

    for filename in filenames:
        # print(filename)
        # print(dirpath)
        # print(dirnames)
        # print(filename)
        file_path = os.path.join(dirpath, filename)
        # file_path = file_path[2:]
        file_path = file_path.replace("\\", "/")
        print(file_path)
        
        file_path_exists = df[df["path"] == file_path].shape[0] > 0 #flag
        # print(df[df["path"] == file_path])
        # print(file_path)
        if file_path_exists:
            
            print(file_path)

            # identifier care
            
            identifier = df.loc[df["path"] == file_path, "Id"].values[0]
            identifier = str(int(identifier))
            
            # label constructor

            action = df.loc[df["path"] == file_path, "action"].values[0]
            object = df.loc[df["path"] == file_path, "object"].values[0]
            label  = action + object


            new_file_path = os.path.join(new_folder_path, identifier + "_" + label + '.wav')

            shutil.copy(file_path, new_file_path)

print(df)
print("Execution ended")
