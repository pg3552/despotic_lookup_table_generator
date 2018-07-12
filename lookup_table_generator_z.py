# NOTES TO DO:

# 1. si the velocity dispersion forced to be virialised?
# If not, need to make sure it is.

from astropy import units
from astropy import constants
import itertools
import numpy as np

import despotic
import copy

########################################################################
# User-settable options
########################################################################


# Set the column density for 3 clouds that we'll study: units are Msun/pc^2
column_density = np.logspace(0., 3., 10) * (units.Msun / (units.pc**2.))

# Set number of radial zones in each cloud
NZONES = 4

# Set up the metallicities
metalgrid = np.linspace(1.5, 0.1, 3)

# Set the nH grid
nhgrid = np.logspace(0.1, 3., 10)

# Set the SFR grid
sfrgrid = np.logspace(0., 3., 10)

# Set the redshift grid
zgrid = np.linspace(0., 5., 11)


# -- Program code ------------------------------------------------------------

ncolumns = column_density.size
nmetals = metalgrid.size
ndens = nhgrid.size
nsfr = sfrgrid.size
nreds = zgrid.size

CO_lines_array = np.zeros((nreds, nmetals, ncolumns, ndens, nsfr, 10))
CI_lines_array = np.zeros((nreds, nmetals, ncolumns, ndens, nsfr, 2))
CII_lines_array = np.zeros((nreds, nmetals, ncolumns, ndens, nsfr))
OI_lines_array = np.zeros([nmetals,ncolumns,ndens,nsfr,2])

H2_abu_array = np.zeros((nreds, nmetals, ncolumns, ndens, nsfr))
HI_abu_array = np.zeros((nreds, nmetals, ncolumns, ndens, nsfr))
CO_abu_array = np.zeros([nmetals,ncolumns,ndens,nsfr])
CI_abu_array = np.zeros([nmetals,ncolumns,ndens,nsfr])
CII_abu_array = np.zeros([nmetals,ncolumns,ndens,nsfr])

CO_intTB_array = np.zeros((nreds, nmetals, ncolumns, ndens, nsfr, 10))
CI_intTB_array = np.zeros((nreds, nmetals, ncolumns, ndens, nsfr, 2))
CII_intTB_array = np.zeros((nreds, nmetals, ncolumns, ndens, nsfr))
OI_intTB_array = np.zeros([nmetals,ncolumns,ndens,nsfr,2])

Tg_array = np.zeros([nmetals,ncolumns,ndens,nsfr])

# Convert the column densities to CGS
mu = 2.33
column_density_cgs = (column_density / (mu * constants.m_p)).cgs

for (nm, nc, nd, nsf, nrs) in itertools.product(
        range(nmetals), range(ncolumns), range(ndens),
        range(nsfr), range(nreds)
        ):

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
    except (despotic.despoticError,ValueError,np.linalg.linalg.LinAlgError,IndexError):
		gmc = copy.deepcopy(gmc_old)

    gmc_old = copy.deepcopy(gmc)

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

    Tg_array[nm,nc,nd,nsf,:] = np.average(gmc.Tg, weights=gmc.mass())

np.savez(
        'high_res_z.npz',
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
        H2_abu_array=H2_abu_array,
        HI_abu_array=HI_abu_array,
        CO_abu_array=CO_abu_array,
        CI_abu_array=CI_abu_array,
        CII_abu_array=CII_abu_array,
		Tg_array=Tg_array
        )
