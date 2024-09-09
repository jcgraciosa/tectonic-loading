import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

### for now only supports the realistic loading type ###
def interpolate_load(step, in_csv, out_csv):

    # requirement is that 1st column should be the position along the axis, 2nd column is the slab pull, and 3rd column is the bending moment
    orig_data = pd.read_csv(in_csv)
    orig_data = orig_data.drop_duplicates(subset = 'Along')

    # do the interplation stuff
    depth = np.asarray(orig_data['Slab length'])
    depth_ref = depth[-1]
    mo_ref = 0.86e17 # these values are from Raghuram's paper
    vo_ref = 0.26e12

    mo_per_km = mo_ref/depth_ref
    vo_per_km = vo_ref/depth_ref

    y_old = np.array(orig_data.iloc[:, 0] - orig_data.iloc[:, 0].min())
    vo_old = vo_per_km*depth
    mo_old = mo_per_km*depth

    vo_func = interp1d(y_old, vo_old, kind = 'cubic')
    mo_func = interp1d(y_old, mo_old, kind = 'cubic')

    y_new = np.arange(0, y_old.max() + step, step) # just make sure this will give something correct
    vo_new = vo_func(y_new)
    mo_new = mo_func(y_new)

    # package results into a dataframe then write to out_csv
    data = np.vstack([y_new, vo_new, mo_new]).T
    out_df = pd.DataFrame(data=data, columns=['Along', 'Vo', 'Mo'])
    out_df.to_csv(out_csv, sep = ',', index = False)
    print("Interpolated load and moment in: ", out_csv)

    return










