from lookup_table_reader import *
import numpy as np

lookup_npzfilename = "/home/desika.narayanan/despotic_lookup_table_generator/high_res.npz"#_fixed_emission_line_width.npz"


column_points = np.asarray([25,50,100]) #Msun/pc^2
nh_points = np.asarray([50,100,1000]) #cm^-3
sfr_points = np.asarray([1,10,20]) #msun/yr
metal_points = np.asarray([1,1,1]) #units of solar

temp_wco_grid,temp_wci_grid,temp_wcii_grid,temp_h2_abu_grid,temp_hi_abu_grid = get_co(lookup_npzfilename,column_points,metal_points,nh_points,sfr_points,intensity=True)

#temp_wco_grid will return a 10 x n_dim array where the 10 dimension
#is the first 10 levels of CO, and the n_dim are the number of
#dimensions of your (e.g.) column_points array.  the units are K-km/s

#wci and wcii are the same but for CI and [CII]

#h2_abu_grid and hi_abu_grid are the H2 and HI abundances such that:

'''
In [7]: temp_h2_abu_grid*2+temp_hi_abu_grid
Out[7]: array([ 0.99997152,  0.99984029,  0.99998034])
'''
