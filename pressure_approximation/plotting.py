import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize
from pressure_determination import fit_press
from constants import *

# Set pyplot parameters, taken from Pauls style.py script.
plt.rcParams['text.latex.preamble'] = [r"\usepackage{lmodern}"]
params = {
    'font.size': 12,
    'font.family': 'DejaVuSans',
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'text.usetex': True,
    
    'figure.dpi': 300,
    'savefig.dpi': 300,
    
    'lines.linewidth': 1.0,
    'lines.dashed_pattern': [3, 2]
}
plt.rcParams.update(params)


def plot_press_prof(x, y, z, data, xlabel='', ylabel='', xtype='cyl', name=0):
    print('Plotting: %s.png' % name)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

    if xtype == 'sph':
        rr = (xx ** 2 + yy ** 2 + zz ** 2) ** 0.5
    elif xtype == 'cyl':
        rr = (xx ** 2 + yy ** 2) ** 0.5
    else:
        rr = (xx ** 2 + yy ** 2) ** 0.5

    rr /= const.parsec / const.centi

    fig = plt.figure()
    ax = fig.gca()
    ax.scatter(
        rr.flatten(),
        data.flatten(),
        c=np.abs(zz).flatten(),
        marker='x',
        s=0.01, rasterized=True
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yscale('log')
    fig.savefig('%s.pdf' % name, format='pdf', dpi=500)


def plot_something(x, y, z, dens, pres_rrz, pres_rzr, pres_zrr):
    plt.close('all')

    fig = plt.figure()
    ax = fig.gca()
    ax.set_xlabel('R [pc]')
    ax.set_ylabel('T [K]')
    fit_press(
        #prof=(pres_rrz + pres_rzr + pres_zrr) / (3. * dens),# * 1e7 * const.gas_constant),
        prof=pres_rrz / (1. * dens),# * 1e7 * const.gas_constant),
        x=x, y=y, z=z, fname='coef_rrz', order=15, ax=ax
    )

    fig.show()

    fig = plt.figure()
    ax = fig.gca()
    ax.set_xlabel('R [pc]')
    ax.set_ylabel('T [K]')
    fit_press(
        #prof=(pres_rrz + pres_rzr + pres_zrr) / (3. * dens),# * 1e7 * const.gas_constant),
        prof=pres_rzr / (1. * dens),# * 1e7 * const.gas_constant),
        x=x, y=y, z=z, fname='coef_rzr', order=15, ax=ax
    )
    fig.show()

    fig = plt.figure()
    ax = fig.gca()
    ax.set_xlabel('R [pc]')
    ax.set_ylabel('T [K]')
    fit_press(
        #prof=(pres_rrz + pres_rzr + pres_zrr) / (3. * dens),# * 1e7 * const.gas_constant),
        prof=pres_zrr / (1. * dens),# * 1e7 * const.gas_constant),
        x=x, y=y, z=z, fname='coef_zrr', order=15, ax=ax
    )
    fig.show()
