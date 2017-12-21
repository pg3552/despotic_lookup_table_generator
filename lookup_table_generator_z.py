#NOTES TO DO:

#1. si the velocity dispersion forced to be virialised?  If not, need to make sure it is.

# Import the despotic library and various standard python libraries
from despotic import cloud,zonedcloud
from despotic.chemistry import NL99_GC
import despotic
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy import constants as constants
#import ipdb,pdb

import pickle
import copy

########################################################################
# User-settable options
########################################################################


#set the column density for 3 clouds that we'll study: units are Msun/pc^2
#column_density = np.array([75,250,1000])* u.Msun/u.pc**2.
column_density = np.linspace(0,3,10)
column_density = 10.**(column_density)*u.Msun/u.pc**2

#set number of radial zones in each cloud
NZONES = 4

metalgrid = np.linspace(1.5,0.1,3) #set up the metallicities 

#set the nH grid
nhgrid = np.linspace(0.1,3,10)
nhgrid = 10.**nhgrid

#set the SFR grid
sfrgrid = np.linspace(0,3,10)
sfrgrid = 10.**sfrgrid

#set the redshift grid
zgrid = np.linspace(0,5,11)

#DEBUG
'''
column_density = np.array([100])*u.Msun/u.pc**2
nhgrid = np.array([100])
sfrgrid = np.array([30])
metalgrid = np.array([1])
zgrid = np.array([0])
'''

##################################################e######################
# Program code
########################################################################

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



ncolumns = len(column_density)
nmetals = len(metalgrid)
ndens = len(nhgrid)
nsfr = len(sfrgrid)
nreds = len(sfrgrid)

obj_list = MultiDimList((nreds,nmetals,ncolumns,ndens,nsfr))
CO_lines_list = MultiDimList((nreds,nmetals,ncolumns,ndens,nsfr,10))
CI_lines_list = MultiDimList((nreds,nmetals,ncolumns,ndens,nsfr,2))
CII_lines_list = MultiDimList((nreds,nmetals,ncolumns,ndens,nsfr))
H2_abu_list = MultiDimList((nreds,nmetals,ncolumns,ndens,nsfr))
HI_abu_list = MultiDimList((nreds,nmetals,ncolumns,ndens,nsfr))

CO_lines_array = np.zeros([nreds,nmetals,ncolumns,ndens,nsfr,10])
CI_lines_array = np.zeros([nreds,nmetals,ncolumns,ndens,nsfr,2])
CII_lines_array = np.zeros([nreds,nmetals,ncolumns,ndens,nsfr])
H2_abu_array = np.zeros([nreds,nmetals,ncolumns,ndens,nsfr])
HI_abu_array = np.zeros([nreds,nmetals,ncolumns,ndens,nsfr])

CO_intTB_array = np.zeros([nreds,nmetals,ncolumns,ndens,nsfr,10])
CI_intTB_array = np.zeros([nreds,nmetals,ncolumns,ndens,nsfr,2])
CII_intTB_array = np.zeros([nreds,nmetals,ncolumns,ndens,nsfr])



#convert the column densities to CGS
mu = 2.33
column_density_cgs = (column_density/(mu*constants.m_p)).cgs

for nm in range(nmetals):
    for nc in range(ncolumns):
        for nd in range(ndens):
            for nsf in range(nsfr):
				for nrs in range(nreds):
                
                
					print '============================='
					print (nm,nc,nd,nsf,nrs)
					print '============================='
                


					#set up the zoned cloud (a radially stratified cloud)
					gmc = zonedcloud(colDen = np.linspace(column_density_cgs[nc].value/NZONES,column_density_cgs[nc].value,NZONES))
                

					#usually we import these via some *.desp file.  Here, we
					#explicitly say these values as they depend on the metallicity
					#as well, and can impact how the CRs and UV radiation can get
					#in.
                
					gmc.sigmaD10   = 2.0e-26  * metalgrid[nm]       # Cross section to 10K thermal radiation, cm^2 H^-1
					gmc.sigmaDPE   = 1.0e-21 * metalgrid[nm]        # Cross section to 8-13.6 eV photons, cm^2 H^-1
					gmc.sigmaDISRF = 3.0e-22  * metalgrid[nm]       # Cross section to ISRF photons, cm^2 H^-1
					gmc.Zdust      = 1.0  * metalgrid[nm]       # Dust abundance relative to solar
					gmc.alphaGD	   = 3.2e-34 * metalgrid[nm]	     # Dust-gas coupling coefficient, erg cm^3 K^-3/2
					gmc.beta   = 2.0							# Dust spectral index
                
                
					gmc.dust.sigma10 = 2.0e-26  * metalgrid[nm]
					gmc.dust.sigmaPE = 1.0e-21 * metalgrid[nm]
					gmc.dust.sigmaISRF = 3.0e-22  * metalgrid[nm]
					gmc.dust.Zd = 1.0  * metalgrid[nm]
					gmc.dust.alphaGD =  3.2e-34 * metalgrid[nm]
					gmc.dust.beta =  2.0
            
            
					#initalise the emitter abundances
                
					gmc.addEmitter('c+',1.e-100)
					gmc.addEmitter('c',2.e-4)
					gmc.addEmitter('o', 4.e-4)
					gmc.addEmitter('co',1.e-100)
                
					gmc.Tcmb = 2.73 * (1 + zgrid[nrs])
                

					#initialise the abundances for H2 and He, and tell the code to
					#extrapolate from the collision tables if you hit a derived
					#temperature outside of the Leiden MolData
					for nz in range(NZONES):
						gmc.comp[nz].xH2 = 0.5
						gmc.comp[nz].xHe = 0.1
						gmc.emitters[nz]['co'].extrap = True
						gmc.emitters[nz]['c+'].extrap = True
						gmc.emitters[nz]['o'].extrap = True
						gmc.emitters[nz]['c'].extrap = True


       


					#================================================================
					#SUBGRID MODEL STUFF FOR CLOUDS
					#================================================================
                
					#we put in turbulent compression after zoned_cloud_properties
					#since we don't actually want to always scale up the gmc.nh
					#(i.e. if tempeq isn't being called) Note, if you call TempEq
					#then you have to put in the argument noClump=True, or
					#alternatively get rid of the manual turbulent compression of
					#densities
                
					gamma = 1.4
					cs = np.sqrt(gamma*constants.k_B/mu/constants.m_p*10.*u.K) #assuming temp of 10K
					alpha_vir = 1.0 # arbitrary value, assuming cloud is virialized
					sigma_vir = np.sqrt(4.0/15.0*np.pi*alpha_vir*constants.G*mu*constants.m_p*column_density_cgs[nc]**2/(nhgrid[nd]/u.cm**3))# assuming spherical cloud
					sigma_vir = max(cs,sigma_vir)
					sigmaNT = np.sqrt(sigma_vir**2-cs**2)
                

					#assign other properties of clouds
                
					SFR = sfrgrid[nsf]
					gmc.nH = nhgrid[nd] 
					gmc.Td = 10
					gmc.Tg = 10
					gmc.rad.TradDust = 10
					gmc.ionRate = 1.e-17*SFR 
					gmc.rad.ionRate = 1.e-17*SFR 
					gmc.chi = 1.*SFR 
					gmc.rad.chi = 1*SFR
       
					gmc.sigmaNT = np.repeat(sigmaNT.cgs.value,NZONES)
					#================================================================
                
                
                
					#actually run the chemical equilibrium model.  this evolves
					#the temperature calcualtion in iteration with the chemistry
					#which is slower, but the most right thing to do as it
					#simultaneously solves for radiative transfer, chemistry and
					#temperature all at once.
					try:
						gmc.setChemEq(network=NL99_GC, evolveTemp = 'iterate', verbose=True)
						gmc.lineLum('co')[0]['lumPerH']
					except (despotic.despoticError,ValueError,np.linalg.linalg.LinAlgError,IndexError):
						gmc = copy.deepcopy(gmc_old)

					gmc_old = copy.deepcopy(gmc)
                                
					#calculate the CO and C+ lines since we really don't want to have to do that later
					CO_lines_array[nrs,nm,nc,nd,nsf,:] = np.array([gmc.lineLum('co')[r]['lumPerH'] for r in range(10)])
					CI_lines_array[nrs,nm,nc,nd,nsf,:] = np.array([gmc.lineLum('c')[r]['lumPerH'] for r in range(2)])
					CII_lines_array[nrs,nm,nc,nd,nsf] =  gmc.lineLum('c+')[0]['lumPerH']

					CO_intTB_array[nrs,nm,nc,nd,nsf,:] = np.array([gmc.lineLum('co')[r]['intTB'] for r in range(10)])
					CI_intTB_array[nrs,nm,nc,nd,nsf,:] = np.array([gmc.lineLum('c')[r]['intTB'] for r in range(2)])
					CII_intTB_array[nrs,nm,nc,nd,nsf] =  gmc.lineLum('c+')[0]['intTB']
                  
					H2_abu_array[nrs,nm,nc,nd,nsf] = np.average(np.array([gmc.chemabundances_zone[n]['H2'] for n in range(NZONES)]),weights=gmc.mass())
					HI_abu_array[nrs,nm,nc,nd,nsf] = np.average(np.array([gmc.chemabundances_zone[n]['H'] for n in range(NZONES)]),weights=gmc.mass())

					#for i in range(10): CO_lines_list.set((nm,nc,nd,nsf,nrs,i),CO_lines[i])
					#CII_lines_list.set((nrs,nm,nc,nd,nsf),CII_lines)
					obj_list.set((nrs,nm,nc,nd,nsf),gmc) #DEBUG

np.savez('high_res_z.npz',redshift = zgrid,column_density = column_density.value,metalgrid = metalgrid,nhgrid = nhgrid,sfrgrid = sfrgrid, CO_lines_array = CO_lines_array, CI_lines_array = CI_lines_array, CII_lines_array = CII_lines_array,CO_intTB_array = CO_intTB_array, CI_intTB_array = CI_intTB_array, CII_intTB_array=CII_intTB_array, H2_abu_array = H2_abu_array, HI_abu_array = HI_abu_array)
'''
filehandler = open("junk.obj","wb")
pickle.dump(obj_list,filehandler)
filehandler.close()

data = [obj_list,CO_lines_list]
with open("high_res.obj","wb") as f:
    pickle.dump((obj_list),f)

'''
