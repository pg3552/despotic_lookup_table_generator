# NOTES TO DO:

# 1. si the velocity dispersion forced to be virialised?
# If not, need to make sure it is.

# Import MPI-related libraries
from mpi4py import MPI
import pnumpy.pnCubeDecomp as CubeDecomp

# Import the despotic library and standard python libraries
from astropy import units
from astropy import constants
import itertools
import numpy as np

import despotic
import copy

########################################################################
# MPI related variables
########################################################################
comm = MPI.COMM_WORLD
status = MPI.Status()
nprocs = comm.Get_size()
rank = comm.Get_rank()
TAG_DECOMP = 0
TAG_GATHER = 1

########################################################################
# User-settable options
########################################################################

# Set number of radial zones in each cloud
NZONES = 8

# Set the column density for 3 clouds that we'll study: units are Msun/pc^2
column_density = np.logspace(0., 3., 10) * (units.Msun / (units.pc**2.))
# Convert the column densities to CGS
mu = 2.33
column_density_cgs = (column_density / (mu * constants.m_p)).cgs

# Set up the metallicities
metalgrid = np.linspace(1.5, 0.1, 3)

# Set the nH grid
nhgrid = np.logspace(0.1, 3., 10)

# Set the SFR grid
sfrgrid = np.logspace(0., 3., 10)

# Set the redshift grid
zgrid = np.linspace(0., 5., 11)


# -- Program code ------------------------------------------------------------

gcolumns = np.arange(column_density.size,dtype=int)
gmetals = np.arange(metalgrid.size,dtype=int)
gdens = np.arange(nhgrid.size,dtype=int)
gsfr = np.arange(sfrgrid.size,dtype=int)
greds = np.arange(zgrid.size,dtype=int)

if rank == 0:
### manager
	ncolumns = column_density.size
	nmetals = metalgrid.size
	ndens = nhgrid.size
	nsfr = sfrgrid.size
	nreds = zgrid.size

	CO_lines_array = np.zeros((nreds, nmetals, ncolumns, ndens, nsfr, 10))
	CI_lines_array = np.zeros((nreds, nmetals, ncolumns, ndens, nsfr, 2))
	CII_lines_array = np.zeros((nreds, nmetals, ncolumns, ndens, nsfr))
	OI_lines_array = np.zeros([nreds,nmetals,ncolumns,ndens,nsfr,2])

	H2_abu_array = np.zeros((nreds, nmetals, ncolumns, ndens, nsfr))
	HI_abu_array = np.zeros((nreds, nmetals, ncolumns, ndens, nsfr))
	CO_abu_array = np.zeros([nreds,nmetals,ncolumns,ndens,nsfr])
	CI_abu_array = np.zeros([nreds,nmetals,ncolumns,ndens,nsfr])
	CII_abu_array = np.zeros([nreds,nmetals,ncolumns,ndens,nsfr])

	CO_intTB_array = np.zeros((nreds, nmetals, ncolumns, ndens, nsfr, 10))
	CI_intTB_array = np.zeros((nreds, nmetals, ncolumns, ndens, nsfr, 2))
	CII_intTB_array = np.zeros((nreds, nmetals, ncolumns, ndens, nsfr))
	OI_intTB_array = np.zeros([nreds,nmetals,ncolumns,ndens,nsfr,2])

	CO_intIntensity_array = np.zeros((nreds, nmetals, ncolumns, ndens, nsfr, 10))
	CI_intIntensity_array = np.zeros((nreds, nmetals, ncolumns, ndens, nsfr, 2))
	CII_intIntensity_array = np.zeros((nreds, nmetals, ncolumns, ndens, nsfr))
	OI_intIntensity_array = np.zeros([nreds,nmetals,ncolumns,ndens,nsfr,2])

	Tg_array = np.zeros([nreds,nmetals,ncolumns,ndens,nsfr])

	## send domain decomposition to workers
	dd = CubeDecomp.CubeDecomp(nprocs=nprocs-1, dims=(nreds,nmetals,ncolumns,ndens,nsfr))
	if dd.isValid() == False:
		print 'Decomposition failed, please use recommended nprocs PLUS 1:'
		print dd.getNumberOfValidProcs()
		comm.Abort()
	for i in range(nprocs-1):
		decomp = dd.getSlab(i)
		comm.send(decomp, dest = i+1, tag = TAG_DECOMP)
		
	## collect results from workers
	for i in range(nprocs-1):
		buff = comm.recv(source=MPI.ANY_SOURCE,tag=TAG_GATHER,status=status) #to-do: use gather instead
		nn = status.Get_source()
		decomp = dd.getSlab(nn)
		
		CO_lines_array[decomp] = buff['CO_lines']
		CI_lines_array[decomp] = buff['CI_lines']
		CII_lines_array[decomp] = buff['CII_lines']
		OI_lines_array[decomp] = buff['OI_lines']
		H2_abu_array[decomp] = buff['H2_abu']
		HI_abu_array[decomp] = buff['HI_abu']
		CO_abu_array[decomp] = buff['CO_abu']
		CI_abu_array[decomp] = buff['CI_abu']
		CII_abu_array[decomp] = buff['CII_abu']
		CO_intTB_array[decomp] = buff['CO_intTB']
		CI_intTB_array[decomp] = buff['CI_intTB']
		CII_intTB_array[decomp] = buff['CII_intTB']
		OI_intTB_array[decomp] = buff['OI_intTB']
		CO_intIntensity_array[decomp] = buff['CO_intIntensity']
		CI_intIntensity_array[decomp] = buff['CI_intIntensity']
		CII_intIntensity_array[decomp] = buff['CII_intIntensity']
		OI_intIntensity_array[decomp] = buff['OI_intIntensity']
		Tg_array[decomp] = buff['Tg']

	## output
	print "Finish collecting data from all processes; save lookup table............"
	np.savez(
		'testintIntensity.npz',
		redshift=zgrid,
		column_density=column_density.value,
		metalgrid=metalgrid,
		nhgrid=nhgrid,
		sfrgrid=sfrgrid,
		CO_lines_array=CO_lines_array,
		CI_lines_array=CI_lines_array,
		CII_lines_array=CII_lines_array,
		CO_intTB_array=CO_intTB_array,
		CI_intTB_array=CI_intTB_array,
		CII_intTB_array=CII_intTB_array,
		CO_intIntensity_array=CO_intIntensity_array,
		CI_intIntensity_array=CI_intIntensity_array,
		CII_intIntensity_array=CII_intItensity_array,
		H2_abu_array=H2_abu_array,
		HI_abu_array=HI_abu_array,
		CO_abu_array=CO_abu_array,
		CI_abu_array=CI_abu_array,
		CII_abu_array=CII_abu_array,
		Tg_array=Tg_array
	)

else:
### workers
	decomp = comm.recv(source=0, tag = TAG_DECOMP)

	nreds = greds[decomp[0]].size
	nmetals = gmetals[decomp[1]].size
	ncolumns = greds[decomp[2]].size
	ndens = gdens[decomp[3]].size
	nsfr = gsfr[decomp[4]].size

	CO_lines_array = np.zeros((nreds, nmetals, ncolumns, ndens, nsfr, 10))
	CI_lines_array = np.zeros((nreds, nmetals, ncolumns, ndens, nsfr, 2))
	CII_lines_array = np.zeros((nreds, nmetals, ncolumns, ndens, nsfr))
	OI_lines_array = np.zeros([nreds,nmetals,ncolumns,ndens,nsfr,2])

	H2_abu_array = np.zeros((nreds, nmetals, ncolumns, ndens, nsfr))
	HI_abu_array = np.zeros((nreds, nmetals, ncolumns, ndens, nsfr))
	CO_abu_array = np.zeros([nreds,nmetals,ncolumns,ndens,nsfr])
	CI_abu_array = np.zeros([nreds,nmetals,ncolumns,ndens,nsfr])
	CII_abu_array = np.zeros([nreds,nmetals,ncolumns,ndens,nsfr])

	CO_intTB_array = np.zeros((nreds, nmetals, ncolumns, ndens, nsfr, 10))
	CI_intTB_array = np.zeros((nreds, nmetals, ncolumns, ndens, nsfr, 2))
	CII_intTB_array = np.zeros((nreds, nmetals, ncolumns, ndens, nsfr))
	OI_intTB_array = np.zeros([nreds,nmetals,ncolumns,ndens,nsfr,2])

	CO_intIntensity_array = np.zeros((nreds, nmetals, ncolumns, ndens, nsfr, 10))
	CI_intIntensity_array = np.zeros((nreds, nmetals, ncolumns, ndens, nsfr, 2))
	CII_intIntensity_array = np.zeros((nreds, nmetals, ncolumns, ndens, nsfr))
	OI_intIntensity_array = np.zeros([nreds,nmetals,ncolumns,ndens,nsfr,2])

	Tg_array = np.zeros([nreds,nmetals,ncolumns,ndens,nsfr])


	for (im, ic, iden, isf, irs) in itertools.product(
			range(nmetals), range(ncolumns), range(ndens),
			range(nsfr), range(nreds)
			):

		nrs, nm, nc, nd, nsf = greds[decomp[0]][0] + irs,\
							   gmetals[decomp[1]][0] + im,\
							   gcolumns[decomp[2]][0] + ic,\
							   gdens[decomp[3]][0] + iden,\
							   gsfr[decomp[4]][0] + isf

		print('=============================')
		print(nm, nc, nd, nsf, nrs)
		print('=============================')

		# Set up the zoned cloud (a radially stratified cloud)
		gmc = despotic.zonedcloud(
				colDen=np.linspace(
					column_density_cgs[nc].value / NZONES,
					column_density_cgs[nc].value,
					NZONES
				)
		)

		# Usually we import these via some *.desp file.  Here, we
		# explicitly say these values as they depend on the metallicity
		# as well, and can impact how the CRs and UV radiation can get
		# in.

		# Cross section to 10K thermal radiation, cm^2 H^-1
		gmc.sigmaD10 = 2e-26 * metalgrid[nm]
		# Cross section to 8-13.6 eV photons, cm^2 H^-1
		gmc.sigmaDPE = 1e-21 * metalgrid[nm]
		# Cross section to ISRF photons, cm^2 H^-1
		gmc.sigmaDISRF = 3e-22 * metalgrid[nm]
		# Dust abundance relative to solar
		gmc.Zdust = 1. * metalgrid[nm]
		# Dust-gas coupling coefficient, erg cm^3 K^-3/2
		gmc.alphaGD = 3.2e-34 * metalgrid[nm]
		# Dust spectral index
		gmc.beta = 2.

		gmc.dust.sigma10 = 2e-26 * metalgrid[nm]
		gmc.dust.sigmaPE = 1e-21 * metalgrid[nm]
		gmc.dust.sigmaISRF = 3e-22 * metalgrid[nm]
		gmc.dust.Zd = 1. * metalgrid[nm]
		gmc.dust.alphaGD = 3.2e-34 * metalgrid[nm]
		gmc.dust.beta = 2.

		# Initalise the emitter abundances
		gmc.addEmitter('c+', 1e-100)
		gmc.addEmitter('c', 2e-4)
		gmc.addEmitter('o', 4e-4)
		gmc.addEmitter('co', 1e-100)

		gmc.Tcmb = 2.73 * (1. + zgrid[nrs])

		# Initialise the abundances for H2 and He, and tell the code to
		# extrapolate from the collision tables if you hit a derived
		# temperature outside of the Leiden MolData
		for nz in range(NZONES):
			gmc.comp[nz].xH2 = 0.5
			gmc.comp[nz].xHe = 0.1
			gmc.emitters[nz]['co'].extrap = True
			gmc.emitters[nz]['c+'].extrap = True
			gmc.emitters[nz]['o'].extrap = True
			gmc.emitters[nz]['c'].extrap = True

		# -- SUBGRID MODEL STUFF FOR CLOUDS  ---------------------
		# we put in turbulent compression after zoned_cloud_properties
		# since we don't actually want to always scale up the gmc.nh
		# (i.e. if tempeq isn't being called) Note, if you call TempEq
		# then you have to put in the argument noClump=True, or
		# alternatively get rid of the manual turbulent compression of
		# densities

		gamma = 1.4
		cs = np.sqrt(
				gamma * constants.k_B / mu /
				constants.m_p * 10. * units.K
				)  # Assuming temp of 10K
		alpha_vir = 1.  # arbitrary value, assuming cloud is virialized
		sigma_vir = np.sqrt(
						4. / 15. * np.pi * alpha_vir * constants.G *
						mu * constants.m_p * column_density_cgs[nc] ** 2. /
						(nhgrid[nd] / units.cm ** 3.)
						)  # Assuming spherical cloud
		sigma_vir = max(cs, sigma_vir)
		sigmaNT = np.sqrt(sigma_vir ** 2. - cs ** 2.)

		# Assign other properties of clouds
		SFR = sfrgrid[nsf]
		gmc.nH = nhgrid[nd]
		gmc.Td = 10.
		gmc.Tg = 10.
		gmc.rad.TradDust = 10.
		gmc.ionRate = 1e-17 * SFR
		gmc.rad.ionRate = 1e-17 * SFR
		gmc.chi = 1. * SFR
		gmc.rad.chi = 1. * SFR

		gmc.sigmaNT = np.repeat(sigmaNT.cgs.value, NZONES)
		# --------------------------------------------------------

		# Actually run the chemical equilibrium model. This evolves
		# the temperature calcualtion in iteration with the chemistry
		# which is slower, but the most right thing to do as it
		# simultaneously solves for radiative transfer, chemistry and
		# temperature all at once.

		try:
			gmc.setChemEq(
				network=despotic.chemistry.NL99_GC,
				evolveTemp='iterate',
				verbose=True
			)
			gmc.lineLum('co')[0]['lumPerH']

			# Calculate the CO and C+ lines
			CO_lines_array[nrs, nm, nc, nd, nsf, :] = np.array([
					gmc.lineLum('co')[r]['lumPerH'] for r in range(10)
					])
			CI_lines_array[nrs, nm, nc, nd, nsf, :] = np.array([
					gmc.lineLum('c')[r]['lumPerH'] for r in range(2)
					])
			CII_lines_array[nrs, nm, nc, nd, nsf] = gmc.lineLum('c+')[0]['lumPerH']
			OI_lines_array[nrs, nm, nc, nd, nsf, :] = np.array([
					gmc.lineLum('o')[r]['lumPerH'] for r in range(2)
					])

			CO_intTB_array[nrs, nm, nc, nd, nsf, :] = np.array([
					gmc.lineLum('co')[r]['intTB'] for r in range(10)
					])
			CI_intTB_array[nrs, nm, nc, nd, nsf, :] = np.array([
					gmc.lineLum('c')[r]['intTB'] for r in range(2)
					])
			CII_intTB_array[nrs, nm, nc, nd, nsf] = gmc.lineLum('c+')[0]['intTB']
			OI_intTB_array[nrs, nm, nc, nd, nsf, :] = np.array([
					gmc.lineLum('o')[r]['intTB'] for r in range(2)
					])

			CO_intIntensity_array[nrs, nm, nc, nd, nsf, :] = np.array([
					gmc.lineLum('co')[r]['intIntensity'] for r in range(10)
					])
			CI_intIntensity_array[nrs, nm, nc, nd, nsf, :] = np.array([
					gmc.lineLum('c')[r]['intIntensity'] for r in range(2)
					])
			CII_intIntensity_array[nrs, nm, nc, nd, nsf] = gmc.lineLum('c+')[0]['intIntensity']
			OI_intIntensity_array[nrs, nm, nc, nd, nsf, :] = np.array([
					gmc.lineLum('o')[r]['intIntensity'] for r in range(2)
					])

			H2_abu_array[nrs, nm, nc, nd, nsf] = np.average(
					np.array([
						gmc.chemabundances_zone[n]['H2'] for n in range(NZONES)
						]),
					weights=gmc.mass()
					)
			HI_abu_array[nrs, nm, nc, nd, nsf] = np.average(
					np.array([
						gmc.chemabundances_zone[n]['H'] for n in range(NZONES)
						]),
					weights=gmc.mass()
					)
			CO_abu_array[nrs, nm, nc, nd, nsf] = np.average(
					np.array([
						gmc.emitters[n]["co"].abundance for n in range(NZONES)
						]),
					weights=gmc.mass()
					)
			CI_abu_array[nrs, nm, nc, nd, nsf] = np.average(
					np.array([
						gmc.emitters[n]["c"].abundance for n in range(NZONES)
						]),
					weights=gmc.mass()
					)
			CII_abu_array[nrs, nm, nc, nd, nsf] = np.average(
					np.array([
						gmc.emitters[n]["c+"].abundance for n in range(NZONES)
						]),
					weights=gmc.mass()
					)

			Tg_array[nrs,nm,nc,nd,nsf] = np.average(gmc.Tg, weights=gmc.mass())

		except (despotic.despoticError,ValueError,np.linalg.linalg.LinAlgError,IndexError):
			CO_lines_array[nrs, nm, nc, nd, nsf, :] = np.array([ -1 for r in range(10)])
			CI_lines_array[nrs, nm, nc, nd, nsf, :] = np.array([ -1 for r in range(2)])
			CII_lines_array[nrs, nm, nc, nd, nsf] = -1
			OI_lines_array[nrs, nm, nc, nd, nsf, :] = np.array([ -1 for r in range(2)])

			CO_intTB_array[nrs, nm, nc, nd, nsf, :] = np.array([ -1 for r in range(10)])
			CI_intTB_array[nrs, nm, nc, nd, nsf, :] = np.array([ -1 for r in range(2)])
			CII_intTB_array[nrs, nm, nc, nd, nsf] = -1
			OI_intTB_array[nrs, nm, nc, nd, nsf, :] = np.array([ -1 for r in range(2)])

			CO_intIntensity_array[nrs, nm, nc, nd, nsf, :] = np.array([ -1 for r in range(10)])
			CI_intIntensity_array[nrs, nm, nc, nd, nsf, :] = np.array([ -1 for r in range(2)])
			CII_intIntensity_array[nrs, nm, nc, nd, nsf] = -1
			OI_intIntensity_array[nrs, nm, nc, nd, nsf, :] = np.array([ -1 for r in range(2)])

			H2_abu_array[nrs, nm, nc, nd, nsf] = -1
			HI_abu_array[nrs, nm, nc, nd, nsf] = -1
			CO_abu_array[nrs, nm, nc, nd, nsf] = -1
			CI_abu_array[nrs, nm, nc, nd, nsf] = -1
			CII_abu_array[nrs, nm, nc, nd, nsf] = -1

			Tg_array[nrs,nm,nc,nd,nsf] = -1

		## send results back to manager process

	buff = {'CO_lines': CO_lines_array,
			'CI_lines': CI_lines_array,
			'CII_lines': CII_lines_array,
			'OI_lines': OI_lines_array,
			'H2_abu': H2_abu_array,
			'HI_abu': HI_abu_array,
			'CO_abu': CO_abu_array,
			'CI_abu': CI_abu_array,
			'CII_abu': CII_abu_array,
			'CO_intTB': CO_intTB_array,
			'CI_intTB': CI_intTB_array,
			'CII_intTB': CII_intTB_array,
			'OI_intTB': OI_intTB_array,
			'CO_intIntensity': CO_intIntensity_array,
			'CI_intIntensity': CI_intIntensity_array,
			'CII_intIntensity': CII_intIntensity_array,
			'OI_intIntensity': OI_intIntensity_array,
			'Tg': Tg_array}
		
	print "Process",rank," complete. Send results back to the manager............"
	comm.send(buff,dest = 0, tag = TAG_GATHER)
