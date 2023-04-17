import os
import shutil

data_path = r'G:\My Drive\01_Postdoc_HMGU\Python_MyScripts\MIA\Git\ChromRings\data\2_Pol_I_II_III'

for root, dirs, files in os.walk(data_path):
    print(root)
    if root.endswith('55 stacks'):
        import pdb; pdb.set_trace()
    if not files:
        continue
    if not dirs:
        continue
    if not root.endswith('Images'):
        continue
    if root.find('stacks') == -1:
        continue
    pos_path = os.path.dirname(root)
    pos_foldername = os.path.basename(pos_path)
    stacks_path = os.path.dirname(pos_path)
    exp_path = os.path.dirname(stacks_path)
    dst_path = os.path.join(exp_path, pos_foldername)
    import pdb; pdb.set_trace()
    # shutil.move(pos_path, dst_path)