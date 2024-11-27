import sys
import os
import inspect
from datetime import datetime
from pprint import pprint

import traceback
from typing import Iterable

def _warn_ask_install_package(commands: Iterable[str], note_txt=''):
    open_str = '='*100
    sep_str = '-'*100
    commands_txt = '\n'.join([f'  {command}' for command in commands])
    text = (
        f'ChromRings needs to run the following commands{note_txt}:\n\n'
        f'{commands_txt}\n\n'
    )
    question = (
        'How do you want to proceed?: '
        '1) Run the commands now. '
        'q) Quit, I will run the commands myself (1/q): '
    )
    print(open_str)
    print(text)
    
    message_on_exit = (
        '[WARNING]: Execution aborted. Run the following commands before '
        f'running ChromRings again:\n\n{commands_txt}\n'
    )
    msg_on_invalid = (
        '$answer is not a valid answer. '
        'Type "1" to run the commands now or "q" to quit.'
    )
    try:
        while True:
            answer = input(question)
            if answer == 'q':
                print(open_str)
                exit(message_on_exit)
            elif answer == '1':
                break
            else:
                print(sep_str)
                print(msg_on_invalid.replace('$answer', answer))
                print(sep_str)
    except Exception as err:
        traceback.print_exc()
        print(open_str)
        print(message_on_exit)

def _run_pip_commands(commands: Iterable[str]):
    import subprocess
    for command in commands:
        try:
            subprocess.check_call([sys.executable, '-m', *command.split()])
        except Exception as err:
            pass

try:
    import cellacdc
except Exception as err:
    print('ChromRings needs to install Cell-ACDC')
    commands = (
        'pip install git+https://github.com/SchmollerLab/Cell_ACDC.git', 
    )
    _warn_ask_install_package(
        commands, note_txt=' (to install Cell-ACDC package)'
    )
    try:
        _run_pip_commands(commands)
    except Exception as err:
        commands = ('pip install cellacdc',)
        _run_pip_commands(commands)
    print('Cell-ACDC installed')

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

if not os.path.exists(data_info_json_path):
    header = '*'*100
    exit(
        f'{header}\n'
        '[ERROR]: The "data_info.json" file was not found.\n\n'
        'Have a look in the folder `ChromRings/examples_data_info_jsons` '
        'for example files.\n\n'
        'If you need help with this, feel free to open an issue on our '
        'GitHub page at the following link:\n\n'
        'https://github.com/ElpadoCan/ChromRings/issues\n\n'
        'Thank you for your patience!\n'
        f'{header}'
    )

current_analysis_path = os.path.join(chromrings_path, 'current_analysis.py')
if not os.path.exists(current_analysis_path):
    with open(current_analysis_path, 'w') as txt:
        txt.write(
"""
NORMALIZE_EVERY_PROFILE = False
NORMALISE_AVERAGE_PROFILE = True # False with 13_nucleolus_nucleus_profile
RESCALE_INTENS_ZERO_TO_ONE = False
NORMALISE_HOW = 'max' # 'sum' # 'max' # 'tot_fluo' # None with 13_nucleolus_nucleus_profile
USE_ABSOLUTE_DIST = False # True goes with 13_nucleolus_nucleus_profile
ZEROIZE_INNER_LAB_EDGE = False
USE_MANUAL_NUCLEOID_CENTERS = False # True False
PLANE = 'xy' # 'xy', 'yz', or 'xz'
LARGEST_NUCLEI_PERCENT = None # 0.2 # None
MIN_LENGTH_PROFILE_PXL = 0 # 9 (goes with 27_muscles_resol_limit) # 0
CONCATENATE_PROFILES = False # True if profiles should go -1.0 ,0, 1.0 where 0 is center
RESAMPLE_BIN_SIZE_DIST = 5

# '28_test_bin_size'
# '24h recovery'
# '27_muscles_resol_limit'
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
batch_name = '0_check_chromrings_profiles' 
"""
        )
        

pwd_path = os.path.dirname(chromrings_path)
data_path = os.path.join(pwd_path, 'data')
tables_path = os.path.join(pwd_path, 'tables')
figures_path = os.path.join(pwd_path, 'figures')

os.makedirs(tables_path, exist_ok=True)
os.makedirs(figures_path, exist_ok=True)

# To run on 15.06.2023: 
# 2, 4, 5, 8, 10 