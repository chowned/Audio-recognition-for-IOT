import os

folder_path = './dsl_data'

objects = os.listdir(folder_path)

# for obj in objects:
#     print(obj)

# for obj in objects:
#     if os.path.isfile(os.path.join(folder_path, obj)):
#         print(obj)

for dirpath, dirnames, filenames in os.walk(folder_path):
    for filename in filenames:
        print(os.path.join(dirpath, filename))
