import pickle
import numpy as np
import scipy
import ipdb,pdb

from scipy.interpolate import LinearNDInterpolator
from itertools import product

def find_nearest(array,value):
    idx = (np.abs(np.array(array)-value)).argmin()
    return idx

class MultiDimList(object):
    def __init__(self, shape):
        self.shape = shape
        self.L = self._createMultiDimList(shape)
    def get(self, ind):
        if(len(ind) != len(self.shape)): raise IndexError()
        return self._get(self.L, ind)
    def set(self, ind, val):
        if(len(ind) != len(self.shape)): raise IndexError()
        return self._set(self.L, ind, val)
    def _get(self, L, ind):
        return self._get(L[ind[0]], ind[1:]) if len(ind) > 1 else L[ind[0]]
    def _set(self, L, ind, val):
        if(len(ind) > 1): 
            self._set(L[ind[0]], ind[1:], val) 
        else: 
            L[ind[0]] = val
    def _createMultiDimList(self, shape):
        return [self._createMultiDimList(shape[1:]) if len(shape) > 1 else None for _ in range(shape[0])]
    def __repr__(self):
        return repr(self.L)

def interpolator(coords, data, point) :
#from this guy: http://stackoverflow.com/questions/14119892/python-4d-linear-interpolation-on-a-rectangular-grid
    dims = len(point)
    indices = []
    sub_coords = []
    for j in xrange(dims) :
        idx = np.digitize([point[j]], coords[j])[0]
        indices += [[idx - 1, idx]]
        sub_coords += [coords[j][indices[-1]]]
    indices = np.array([j for j in product(*indices)])
    sub_coords = np.array([j for j in product(*sub_coords)])
    sub_data = data[list(np.swapaxes(indices, 0, 1))]
    li = LinearNDInterpolator(sub_coords, sub_data)
    return li([point])[0]




def get_co(picklename,npzname,column_points,metal_points,nh_points,sfr_points,intensity=False):



    with open(picklename,"rb") as f:
        obj_list = pickle.load(f)
    
    data = np.load(npzname)
    column_density = data['column_density']
    metalgrid = data['metalgrid']
    nhgrid = data['nhgrid']
    sfrgrid = data['sfrgrid']

    if intensity == False:
        CO_lines_array = data['CO_lines_array']
        CII_lines_array = data['CII_lines_array']
    else:
        CO_lines_array = data['CO_intTB_array']
        CII_lines_array = data['CII_intTB_array']


    #mask out infs and nans
    CO_lines_array = np.nan_to_num(CO_lines_array)
    CII_lines_array = np.nan_to_num(CII_lines_array)

    #floor and ceiling all the sph particle points
    w_column_min = np.where(column_points < np.min(column_density))[0]
    w_column_max = np.where(column_points > np.max(column_density))[0]
    w_metal_min = np.where(metal_points < np.min(metalgrid))[0]
    w_metal_max = np.where(metal_points > np.max(metalgrid))[0]
    w_nh_min = np.where(nh_points < np.min(nhgrid))[0]
    w_nh_max = np.where(nh_points > np.max(nhgrid))[0]
    w_sfr_min = np.where(sfr_points < np.min(sfrgrid))[0]
    w_sfr_max = np.where(sfr_points > np.max(sfrgrid))[0]

    if len(w_column_min) > 0: column_points[w_column_min] = np.min(column_density)*1.1
    if len(w_column_max) > 0: column_points[w_column_max] = np.max(column_density)*0.9
    if len(w_metal_min) > 0: metal_points[w_metal_min] = np.min(metalgrid)*1.1
    if len(w_metal_max) > 0: metal_points[w_metal_max] = np.max(metalgrid)*0.9
    if len(w_nh_min) > 0: nh_points[w_nh_min] = np.min(nhgrid)*1.1
    if len(w_nh_max) > 0: nh_points[w_nh_max] = np.max(nhgrid)*0.9
    if len(w_sfr_min) > 0: sfr_points[w_sfr_min] = np.min(sfrgrid)*1.1
    if len(w_sfr_max) > 0: sfr_points[w_sfr-max] = np.max(sfrgrid)*0.9
                                                                 
    interpolated_co_lines_array = np.zeros([len(column_points),10])
    for i in range(len(column_points)):
        
        
        #point = np.array([column_points[i],metal_points[i],nh_points[i],sfr_points[i]])
        #interpolated_co_lines = interpolator((column_density,metalgrid,nhgrid,sfrgrid),CO_lines_array,point)

        point = np.array([metal_points[i],column_points[i],nh_points[i],sfr_points[i]])
        interpolated_co_lines = interpolator((metalgrid,column_density,nhgrid,sfrgrid),CO_lines_array,point)
        interpolated_co_lines_array[i,:] = interpolated_co_lines

    return interpolated_co_lines_array

