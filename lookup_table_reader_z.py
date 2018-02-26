import pickle
import numpy as np
import scipy
#import ipdb,pdb

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


def interpolator_qlinear(coords, data, point):
# vectorized 5-D quntilinear interpolation over structured grid
	
	# load the structured grid
	par1 = point[0]
	par10 = coords[0]
	par2 = point[1]
	par20 = coords[1]
	par3 = point[2]
	par30 = coords[2]
	par4 = point[3]
	par40 = coords[3]
	par5 = point[4]
	par50 = coords[4]
	
	varshape = par1.shape
	value21 = np.zeros(varshape)
	value20 = np.zeros(varshape)
	value31 = np.zeros(varshape)
	value30 = np.zeros(varshape)
	value41 = np.zeros(varshape)
	value40 = np.zeros(varshape)
	value51 = np.zeros(varshape)
	value50 = np.zeros(varshape)

	# locate points 
	index_par1max = len(par10)-2
	index_par2max = len(par20)-2
	index_par3max = len(par30)-2
	index_par4max = len(par40)-2
	index_par5max = len(par50)-2
	index1 = np.fmin(index_par1max,np.fmax(0,np.digitize(par1,par10)-1))
	index2 = np.fmin(index_par2max,np.fmax(0,np.digitize(par2,par20)-1))
	index3 = np.fmin(index_par3max,np.fmax(0,np.digitize(par3,par30)-1))
	index4 = np.fmin(index_par4max,np.fmax(0,np.digitize(par4,par40)-1))
	index5 = np.fmin(index_par5max,np.fmax(0,np.digitize(par5,par50)-1))
	
	#interpolation 
	for p in [1,2]:
		for q in [1,2]:
			for w in [1,2]:
				for v in [1,2]:
					# over par5
					slope = (data[p+index1-1,q+index2-1,w+index3-1,v+index4-1,index5+1]-data[p+index1-1,q+index2-1,w+index3-1,v+index4-1,index5])/(par50[index5+1]-par50[index5])
					if v == 1:
						value50 = (par5-par50[index5])*slope+data[p+index1-1,q+index2-1,w+index3-1,v+index4-1,index5]
					else:
						value51 = (par5-par50[index5])*slope+data[p+index1-1,q+index2-1,w+index3-1,v+index4-1,index5]
				# over par4
				slope = (value51-value50)/(par40[index4+1]-par40[index4])
				if w == 1:
					value40 = (par4-par40[index4])*slope+value50
				else:
					value41 = (par4-par40[index4])*slope+value50
			# over par3
			slope = (value41-value40)/(par30[index3+1]-par30[index3])
			if q == 1:
				value30 = (par3-par30[index3])*slope+value40
			else:
				value31 = (par3-par30[index3])*slope+value40
		# over par2
		slope = (value31-value30)/(par20[index2+1]-par20[index2])
		if p == 1:
			value20 = (par2-par20[index2])*slope+value30
		else:
			value21 = (par2-par20[index2])*slope+value30
	# over par1
	slope = (value21-value20)/(par10[index1+1]-par10[index1])
	value = (par1-par10[index1])*slope+value20
	return value



def get_co(npzname,redshift_points,column_points,metal_points,nh_points,sfr_points,intensity=False,log_input=False,log_output=False):

	# input: 
	# npzname: [char] name of tje lookup table, containing grid data (e.g. CO intensity) as a function of grid coordinates (metal,nH,sfr,colDen)
	# redshift,column_points,metal_points,nh_points,sfr_points: [ndarray] must be on linear scale!
	#
	# output: 
	# CO, CI, CII line intensity, H2, H abundance [ndarray]

	# log_input/log_output:
	#	- False: input grid coordinates remain on linear scale/grid data remain on linear scale
	#	- True: input grid coordinates are transformed to log scale (except metallicity)/grid data are transformed to log scale (e.g. CO intensity -> log(CO intensity))
    
	data = np.load(npzname)
	column_density = data['column_density']
	redshift = data['redshift']
	metalgrid = data['metalgrid']
	nhgrid = data['nhgrid']
	sfrgrid = data['sfrgrid']

	if intensity == False:
		CO_lines_array = data['CO_lines_array']
		CI_lines_array = data['CI_lines_array']
		CII_lines_array = data['CII_lines_array']
	else:
		CO_lines_array = data['CO_intTB_array']
		CI_lines_array = data['CI_intTB_array']
		CII_lines_array = data['CII_intTB_array']
	
	HI_abu_array = data['HI_abu_array']
	H2_abu_array = data['H2_abu_array']


    #mask out infs and nans
	CO_lines_array = np.nan_to_num(CO_lines_array)
	CI_lines_array = np.nan_to_num(CI_lines_array)
	CII_lines_array = np.nan_to_num(CII_lines_array)
	H2_abu_array = np.nan_to_num(H2_abu_array)
	HI_abu_array = np.nan_to_num(HI_abu_array)

    #floor and ceiling all the sph particle points
	w_redshift_min = np.where(redshift_points < np.min(redshift))[0]
	w_redshift_max = np.where(redshift_points > np.max(redshift))[0]
	w_column_min = np.where(column_points < np.min(column_density))[0]
	w_column_max = np.where(column_points > np.max(column_density))[0]
	w_metal_min = np.where(metal_points < np.min(metalgrid))[0]
	w_metal_max = np.where(metal_points > np.max(metalgrid))[0]
	w_nh_min = np.where(nh_points < np.min(nhgrid))[0]
	w_nh_max = np.where(nh_points > np.max(nhgrid))[0]
	w_sfr_min = np.where(sfr_points < np.min(sfrgrid))[0]
	w_sfr_max = np.where(sfr_points > np.max(sfrgrid))[0]

	if len(w_redshift_min) > 0: redshift_points[w_redshift_min] = np.min(redshift)*1.1
	if len(w_redshift_max) > 0: redshift_points[w_redshift_max] = np.max(redshift)*0.9
	if len(w_column_min) > 0: column_points[w_column_min] = np.min(column_density)*1.1
	if len(w_column_max) > 0: column_points[w_column_max] = np.max(column_density)*0.9
	if len(w_metal_min) > 0: metal_points[w_metal_min] = np.min(metalgrid)*1.1
	if len(w_metal_max) > 0: metal_points[w_metal_max] = np.max(metalgrid)*0.9
	if len(w_nh_min) > 0: nh_points[w_nh_min] = np.min(nhgrid)*1.1
	if len(w_nh_max) > 0: nh_points[w_nh_max] = np.max(nhgrid)*0.9
	if len(w_sfr_min) > 0: sfr_points[w_sfr_min] = np.min(sfrgrid)*1.1
	if len(w_sfr_max) > 0: sfr_points[w_sfr_max] = np.max(sfrgrid)*0.9
                                                                
	interpolated_co_lines_array = np.zeros([len(column_points),10])
	interpolated_ci_lines_array = np.zeros([len(column_points),2])
	interpolated_cii_lines_array = np.zeros([len(column_points)])
	interpolated_h2_abu_array = np.zeros([len(column_points)])
	interpolated_hi_abu_array = np.zeros([len(column_points)])
    
	
	if log_input == True:
		point = (redshift_points,metal_points,np.log10(column_points),np.log10(nh_points),np.log10(sfr_points))
		coords = (redshift,metalgrid,np.log10(column_density),np.log10(nhgrid),np.log10(sfrgrid))
	else:
		point = (redshift_points,metal_points,column_points,nh_points,sfr_points)
		coords = (redshift,metalgrid,column_density,nhgrid,sfrgrid)

	if log_output == True:
		interpolated_co_lines_array = 10**np.array([interpolator_qlinear(coords,np.log10(CO_lines_array[:,:,:,:,:,r]),point) for r in range(10)])
		interpolated_ci_lines_array = 10**np.array([interpolator_qlinear(coords,np.log10(CI_lines_array[:,:,:,:,:,r]),point) for r in range(2)])
		interpolated_cii_lines_array = 10**interpolator_qlinear(coords,np.log10(CII_lines_array),point)
		interpolated_h2_abu_array = 10**interpolator_qlinear(coords,np.log10(H2_abu_array),point)
		interpolated_hi_abu_array = 10**interpolator_qlinear(coords,np.log10(HI_abu_array),point)
	else:
		interpolated_co_lines_array = np.array([interpolator_qlinear(coords,CO_lines_array[:,:,:,:,:,r],point) for r in range(10)])
		interpolated_ci_lines_array = np.array([interpolator_qlinear(coords,CI_lines_array[:,:,:,:,:,r],point) for r in range(2)])
		interpolated_cii_lines_array = interpolator_qlinear(coords,CII_lines_array,point)
		interpolated_h2_abu_array = interpolator_qlinear(coords,H2_abu_array,point)
		interpolated_hi_abu_array = interpolator_qlinear(coords,HI_abu_array,point)

	return interpolated_co_lines_array,interpolated_ci_lines_array,interpolated_cii_lines_array,2.0*interpolated_h2_abu_array,interpolated_hi_abu_array
