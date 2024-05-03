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
NORMALISE_HOW = 'max' # 'sum' # 'max' # 'tot_fluo' # None
USE_ABSOLUTE_DIST = False
USE_MANUAL_NUCLEOID_CENTERS = True
PLANE = 'xy' # 'xy', 'yz', or 'xz'

# '25_hypoderm', '26_muscles'
# '22_Pol_I_12h', '23_Pol_I_24h'
# '20_WT_two_osmolarities' '21_fast_on_plate_fed_in_liquid'
# '16_cold_shock', '17_heat_shock', '18_Actinomycin', '19_AMPK'
# '15_muscles_fed_vs_starved_histone'
# '13_nucleolus_nucleus_profile', '14_WT_raga1_with_degraded_Pol_I'
# '10_Tir_1', '11_hypoderm', '12_adults'
# '9_raga1', '8_raga1_all_categories'
# '7_WT_starved_vs_fed_histone', '6_WT_fed_DNA_vs_histone'
# '5_WT_starved_DNA_vs_histone', '4_WT_refed'
# '3_Daf15' '2_Pol_I_II_III', '1_test_3D_vs_2D' 
batch_name = '9_raga1' 

# To run on 15.06.2023: 
# 2, 4, 5, 8, 10 