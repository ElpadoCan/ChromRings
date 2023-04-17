import os

data_path = r'G:\My Drive\01_Postdoc_HMGU\Python_MyScripts\MIA\Git\ChromRings\data\2_Pol_I_II'

for root, dirs, files in os.walk(data_path):
    if not files:
        continue
    if not dirs:
        continue
    if not root.endswith('Images'):
        continue
    for file in files:
        file_path = os.path.join(root, file)
        if file.endswith('metadata.csv'):
            with open(file_path, 'r') as csv:
                metadata = csv.read()
                metadata = metadata.replace('w2SDC405', 'SDC405')
            with open(file_path, 'w') as csv:
                csv.write(metadata)
        if file.endswith('w2SDC405.tif'):
            new_filename = file.replace('w2SDC405', 'SDC405')
            new_filepath = os.path.join(root, new_filename)
            os.rename(file_path, new_filepath)