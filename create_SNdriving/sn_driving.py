# Getting SN driving files
import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt


# Conversion factors
m_solar = 1.988e33
pc2cm = const.parsec / const.centi
catalan_const = 0.915965594177219

# Set seed for reproducibility
#seed = 12345
#np.random.seed(seed)

# Parameters to vary
gas_mass = 5e6 * m_solar
a_sc = 250. * pc2cm
b_sc = 100. * pc2cm

# Density floor
rholim = 1e-28
# Only consider disk
only_disk_r = True
only_disk_z = True

# Maximum radius
rsize = 4e3
# Maximum height above midplane
zsize = 4e3

# Central density
rho0 = gas_mass / (8 * np.pi * a_sc**2 * b_sc * catalan_const)
# Conversion from g cm^-2 to Msun pc^-2
conv = (const.parsec/const.centi)**2 / 1.988e33

# Refinement level of grid
lvl = 9
# Number of cells at given refinement level
nrcells = 2**(lvl-1+3-1)

# Bin width in radius and height
dr = rsize / nrcells
dz = rsize / nrcells

# Scale height in Z for random distribution
scale_height = 50 * const.parsec / const.centi

# Determine cell radial position [pc]
rr = np.linspace(dr/2., rsize+dr/2., nrcells, endpoint=False)
# Determine cell heights [pc]
zz = np.linspace(dz/2., rsize+dz/2., nrcells, endpoint=False)

# Determine Gas surface density
# Cutoff when hitting density floor
sig = np.maximum(rho0 / np.cosh(rr * pc2cm / a_sc), rholim)
if only_disk_r:
    sig[sig == rholim] = 0.
sig = np.maximum(sig[:, None] / np.cosh(zz[None, :] * pc2cm / b_sc)**2, rholim)
if only_disk_z:
    sig[sig == rholim] = 0.

sig = sig.sum(axis=1)
sig *= dz * pc2cm * 2

#with open('gas_surface_density.dat', 'w+') as f:
#    f.write('# r [pc]\tsigma [Msun/pc^2]\n')
#    for i in range(sig.size):
#        f.write('%1.9e\t%1.9e\n' % (rr[i], sig[i]*conv))

# Determine SFR surface density
# Msun kpc^-2 yr^-1
sigsfr = 2.5e-4 * (sig*conv)**1.4
# Determine Annuli area
# pc^2
annuli = np.pi * ((rr+dr/2.)**2 - (rr-dr/2.)**2)
# SNR
# Myr^-1
snr = sigsfr * annuli / 120

# Downscale SNR
snr /= 10.

# Total Nr SNe per Myr
snr_tot = snr.sum()

# Total time to consider [Myr]
time = 10.
# sub timestep
dtstep = 1e3
# Times of SNe in each radial bin
sne_rad = [[] for i in range(nrcells)]
for i in range(nrcells):
    # Sub sample
    ne = int(dtstep * time)
    t = np.arange(ne, dtype=float)[np.random.rand(ne) <= snr[i] / dtstep] / dtstep
    t += np.random.rand(t.size) - 0.5
    sne_rad[i].extend(t.tolist())

# Collect info for all SNe
# Time [s], posx [cm], posy [cm], posz [cm], ener [erg]
sn_info = [[] for i in range(5)]
for i in range(nrcells):
    # Count number of SNe within radial bin
    nr_sne_cur = len(sne_rad[i])
    # Add all the times in [s] to end array
    sn_info[0].extend((np.asarray(sne_rad[i]) * const.mega * const.year).tolist())
    # Determine for each SN a random angle
    phi = np.random.rand(nr_sne_cur) * np.pi * 2
    # Determine a position within the radial and convert to [cm]
    dr_random = (np.random.rand(nr_sne_cur) - 0.5) * dr
    x = (rr[i] + dr_random) * np.cos(phi) * const.parsec / const.centi
    dr_random = (np.random.rand(nr_sne_cur) - 0.5) * dr
    y = (rr[i] + dr_random) * np.sin(phi) * const.parsec / const.centi
    # Use Normal distribution to determine height
    z = np.random.normal(scale=scale_height, size=nr_sne_cur)
    # Add new positions to data output
    sn_info[1].extend(x.tolist())
    sn_info[2].extend(y.tolist())
    sn_info[3].extend(z.tolist())
    # Each SN is 10^51 erg
    sn_info[4].extend([1e51] * nr_sne_cur)

# Create last entry of file
# as loop always looks for a later entry in SNdriving.F90
sn_info[0].append(1e+20)
sn_info[1].append(0e+00)
sn_info[2].append(0e+00)
sn_info[3].append(0e+00)
sn_info[4].append(0e+00)

sn_info = np.asarray(sn_info)
sn_info = sn_info[:, np.argsort(sn_info[0])]
sn_info = sn_info[:, sn_info[0] >= 0.0]

nr_sne = sn_info.shape[1]

np.savetxt(
    fname='SNdriving_info.dat',
    X=sn_info.T,
    fmt='%+1.8e', comments='',
    header='# time, posx, posy, posz, ESN (cgs units)\n\t%i' % nr_sne
)



####### Plotting
# Radial profile of Sig_gas, Sig_SFR, SNR
fig = plt.figure()
ax = fig.gca()
ax.plot(rr, sig*conv, label=r'$\Sigma_\mathrm{gas}$ [M$_\odot$ kpc$^{-2}$ yr$^{-1}$]')
ax.plot(rr, sigsfr, label=r'$\Sigma_\mathrm{SFR}$ [M$_\odot$ pc$^{-2}$ Myr$^{-1}$]')
ax.plot(rr, snr, label=r'SNR [Myr$^{-1}$]')
ax.axhline(snr_tot, c='k', ls=':')
ax.text(
    rsize, snr_tot*0.95,
    s=r'SNR$_\mathrm{tot}$ = %3.2f' % snr_tot,
    verticalalignment='top',
    horizontalalignment='right'
)
ax.set_xlabel('R [pc]')
ax.legend()
ax.set_yscale('log')
fig.show()

# Distribution of SNe (face on)
fig = plt.figure()
ax = fig.gca()
sc = ax.scatter(
    sn_info[1] / const.parsec * const.centi,
    sn_info[2] / const.parsec * const.centi,
    c=sn_info[0]/const.mega / const.year, s=0.1
)
ax.set_xlim(-4e3, 4e3)
ax.set_ylim(-4e3, 4e3)
ax.set_xlabel('X [pc]')
ax.set_ylabel('Y [pc]')
plt.colorbar(sc, label='time [Myr]')
fig.show()

# Distribution of SNe (edge on)
fig = plt.figure()
ax = fig.gca()
sc = ax.scatter(
    sn_info[1] / const.parsec * const.centi,
    sn_info[3] / const.parsec * const.centi,
    c=sn_info[0]/const.mega / const.year, s=0.1
)
ax.set_xlabel('X [pc]')
ax.set_ylabel('Z [pc]')
ax.set_xlim(-4e3, 4e3)
ax.set_ylim(-4e3, 4e3)
plt.colorbar(sc, label='time [Myr]')
fig.show()

plt.show()
