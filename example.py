# temp_wco_grid will return a 10 x n_dim array where the 10 dimension
# is the first 10 levels of CO, and the n_dim are the number of
# dimensions of your (e.g.) column_points array.  the units are K-km/s.
# If you set intensity=False then the output will be erg/s/H

# wci and wcii are the same but for CI and [CII]

import numpy as np
import itertools
from lookup_table_reader_z import TableReader
from lookup_table_reader_z_original import get_co

lookup_npzfilename = "./high_res_abu_z-original.npz"

redshift_points = np.asarray([0., 1., 1.])
column_points = np.asarray([25, 50, 100])  # Msun/pc^2
nh_points = np.asarray([50, 100, 1000])  # cm^-3
sfr_points = np.asarray([1, 10, 20])  # msun/yr
metal_points = np.asarray([1, 1, 1])  # units of solar


# -- updated method ---------------------------------------------------------
table = TableReader(lookup_npzfilename)

table.limitsMode = "error"  # raises error
# table.limitsMode = "clip"  # caps values to the limits using np.clip
# table.limitsMode = "value", np.nan  # fills in a constant value
# table.limitsMode = "leave"  # do nothing

# if values exceeding limits
table.copyPoints = True  # copy array and update these values
# table.copyPoints = False  # update original input array

# -- comparision of new method to old ----------------------------------------
for i, li, lo in itertools.product([True, False], repeat=3):

    # -- new method --
    output_new = table.getValues(
            redshift_points, column_points,
            metal_points, nh_points, sfr_points,
            intensity=i, log_input=li, log_output=lo
            )
    # output_new is a dictionary, but for comparision the output
    # is converted into a list in the same order as output_original
    list_new_output = [
            output_new["co"], output_new["ci"], output_new["cii"],
            output_new["h2"], output_new["hi"]
            ]

    # -- original method --
    output_original = get_co(
            lookup_npzfilename,
            redshift_points, column_points,
            metal_points, nh_points, sfr_points,
            intensity=i, log_input=li, log_output=lo
            )

    # -- comparision --
    print("intensity, log_input, log_output: ", i, li, lo)
    for i in range(len(output_new)):
        print("Same output: ", np.allclose(
                list_new_output[i], output_original[i],
                rtol=0., atol=1e-10, equal_nan=True
                ))
