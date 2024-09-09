"""
Use this to run script to run the numerical models in the Monarch server!!!

"""
import time
import numpy as np
import sys
import json
sys.path.append('./')

from pathlib import Path

import plate_model3
#import general_plate
import stress_computer
import fourth_der_computer

# configuration in here
use_cntr = True
conf_file = Path('../config/test_config_multi.json')

# output paths in here
out_dir = Path('../../new_results/sumatra_load/el_area_sc_sstvty_E') # NOTE: replace
deflection_fname = 'el_3d_deflect'
deflection_25D_fname = 'el_2d_deflect'
var_name = 'E' # NOTE: replace with thk, E or filt

# NOTE: replace accordingly
# select one - replace
thk_arr = np.array([1e10, 5e10])
#thk_arr = np.array([10e3, 20e3, 30e3, 40e3])
# thk_arr = np.array(["../indata/area_scaled_vomo/filt_area_sc_vomo_200.csv",
#     "../indata/area_scaled_vomo/filt_area_sc_vomo_400.csv",
#     "../indata/area_scaled_vomo/filt_area_sc_vomo_600.csv",
#     "../indata/area_scaled_vomo/filt_area_sc_vomo_800.csv",
#     "../indata/area_scaled_vomo/filt_area_sc_vomo_1000.csv"])
#    "../indata/area_scaled_vomo/area_sc_pull_and_moment.csv"])

#thk_arr = np.array(["../indata/depth_scaled_vomo/pull_and_moment.csv",
#    "../indata/depth_scaled_vomo/filt_orig_vomo_200.csv",
#    "../indata/depth_scaled_vomo/filt_orig_vomo_400.csv",
#    "../indata/depth_scaled_vomo/filt_orig_vomo_600.csv",
#    "../indata/depth_scaled_vomo/filt_orig_vomo_800.csv",
#    "../indata/depth_scaled_vomo/filt_orig_vomo_1000.csv"])

cntr = 0
for thk in thk_arr:

    print('variable value: ', thk)
    with open(conf_file) as json_data_file:
        data = json.load(json_data_file)

    # replace
    # NOTE: replace accordingly
    data["plate"]["modulus"] = thk # replace with the desired modulus value
    #data["plate"]["vomo_file"] = thk # replace with the desired vomo filename - for filtering
    #data["plate"]["thickness"] = thk # replace with the desired thickness value

    with open(conf_file, 'w') as json_data_file:
        json.dump(data, json_data_file)

    if use_cntr:
        suffix = cntr
        cntr += 1
    else:
        suffix = thk

    out_3d_fname = deflection_fname + '_' + var_name + '_' + str(suffix) + '.hdf5'
    out_2d_fname = deflection_25D_fname + '_' + var_name + '_' +str(suffix) + '.hdf5'
    deflection_out = out_dir/out_3d_fname
    deflection_25D_out = out_dir/out_2d_fname

    # entire process in here
    plate_model = plate_model3.ScaledPlate(conf_file, pl_type = 'elastic', gen_a_matrix = False)
    print('Filling matrix...')
    st = time.time()
    plate_model.fill_matrix()
    ed = time.time()
    elapsed = (ed - st)/60
    print('Filling matrix done after (minutes): ', elapsed)

    print('Solving for deflection...')
    st = time.time()
    plate_model.solve_spsolve()

    plate_model.solve_25D()
    plate_model.rescale_out()
    plate_model.write_out(deflection_out, 0)
    plate_model.write_out(deflection_25D_out, 1)
    ed = time.time()
    elapsed = (ed - st)/60
    print('Solving for deflection done after minutes: ', elapsed)

