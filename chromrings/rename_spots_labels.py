import os
import shutil

chromrings_path = os.path.dirname(os.path.abspath(__file__))
pwd_path = os.path.dirname(chromrings_path)
data_path = os.path.join(pwd_path, 'data')

root_path = os.path.join(data_path, '13_nucleolus_nucleus_profile')

for root, dirs, files in os.walk(root_path):
    if not root.endswith('Images'):
        continue
    for file in files:
        file_path = os.path.join(root, file)
        if file.endswith('spot_labels.npz'):
            new_filename = file.replace('spot_labels.npz', 'segm_nucleolus.npz')
            new_filepath = os.path.join(root, new_filename)
            shutil.copy(file_path, new_filepath)
            print(f'Copied "{file_path}"')