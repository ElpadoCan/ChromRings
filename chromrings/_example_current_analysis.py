
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
CONCATENATE_PROFILES = True # True if profiles should go -1.0 ,0, 1.0 where 0 is center
RESAMPLE_BIN_SIZE_DIST = 10
AUTOMATICALLY_SKIP_POS_WITHOUT_ALL_FILES = False # True

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
batch_name = '28_test_bin_size' 
