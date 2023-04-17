import sys
import os
import inspect
from datetime import datetime
from pprint import pprint

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def printl(*objects, pretty=False, is_decorator=False, **kwargs):
    # Copy current stdout, reset to default __stdout__ and then restore current
    current_stdout = sys.stdout
    sys.stdout = sys.__stdout__
    timestap = datetime.now().strftime('%H:%M:%S')
    currentframe = inspect.currentframe()
    outerframes = inspect.getouterframes(currentframe)
    idx = 2 if is_decorator else 1
    callingframe = outerframes[idx].frame
    callingframe_info = inspect.getframeinfo(callingframe)
    filpath = callingframe_info.filename
    filename = os.path.basename(filpath)
    print_func = pprint if pretty else print
    print('*'*30)
    print(f'{timestap} - File "{filename}", line {callingframe_info.lineno}:')
    print_func(*objects, **kwargs)
    print('='*30)
    sys.stdout = current_stdout

chromrings_path = os.path.dirname(os.path.abspath(__file__))
data_info_json_path = os.path.join(chromrings_path, 'data_info.json')
pwd_path = os.path.dirname(chromrings_path)
data_path = os.path.join(pwd_path, 'data')
tables_path = os.path.join(pwd_path, 'tables')
figures_path = os.path.join(pwd_path, 'figures')

NORMALIZE_EVERY_PROFILE = False
NORMALISE_AVERAGE_PROFILE = True
NORMALISE_HOW = 'max' # 'sum' # 'max' # 'tot_fluo'

batch_name = '1_test_3D_vs_2D' # '3_Pol_I_III_raga1_Daf15' # '2_Pol_I_II'