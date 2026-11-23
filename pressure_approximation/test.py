import sys
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.colors import LogNorm
import scipy.constants as const

import numpy as np
import h5py
import flash_amr_tools as amr_tools

from integration_amr import (
    integrate_amr,
    fill_ghostcells
)

from external_potentials_amr import (
    get_dm_acc_amr,
    get_stellar_acc_amr,
    get_rotation_acc_amr,
    update_rotation_vel_amr
)

from io_amr import (
    create_new_hdf,
    get_new_temp,
    get_new_ener,
    get_new_eint,
    update_all_parent_blocks,
    mirror_data,
)

from constants import m_solar, pc2cm

# fname = sys.argv[1]
fname = '../../../simulation_snapshots/DwarfGal_hdf5_chk_0000'
fname = "/home/nuernberger/tmp/DwarfGal_new/DwarfGal_hdf5_plt_cnt_0000"
# pres_init = float(sys.argv[2])
pres_init = 1e-15
temp_init = 1e5

xmin = []
xmax = []

# Get all blocks within box xmin, xmax
if xmax is None:
    xmax = []
if xmin is None:
    xmin = []

blist, brefs, bns = amr_tools.get_true_blocks(fname, xmin, xmax)

pf = h5py.File(fname, 'r')
keys = pf.keys()

# Get density [g cm^-3]
dens = pf['dens'][()]
pres = pf['pres'][()]
temp = pf['temp'][()]

pres_init = pres[0, 0, 0, 0] * temp_init / temp[0, 0, 0, 0]

del pres, temp

# Get bounding box of each block [cm]
bbox = pf['bounding box'][()]

# Get block center coordinates [cm]
coord = pf['coordinates'][()]

# Get gid
gid = pf['gid'][()]

# Get gravitational acceleration in x, y and z
# [cm s^-2]
gacx = pf['gacx'][()]
gacy = pf['gacy'][()]
gacz = pf['gacz'][()]

ext_acc = False
gexx, gexy, gexz = 0.0, 0.0, 0.0

# Check if external accelerations exists
if 'gexx' in keys and 'gexx' in keys and 'gexx' in keys:
    gexx = pf['gexx'][()]
    gexy = pf['gexy'][()]
    gexz = pf['gexz'][()]
    if np.any(gexx != 0.0) or np.any(gexy != 0.0) or np.any(gexz != 0.0):
        ext_acc = True

# External accelerations cannot be taken from checkpoint file
# Derive them internally
if not ext_acc:
    param_dict = dict(pf['real runtime parameters'][()])
    
    # Get acceleration from dark matter potential
    # mass_dm = 1e12
    tmp_key = ('%-80s' % 'sim_m0').encode('utf-8')
    mass_dm = param_dict[tmp_key] / m_solar
    
    gexx_dm, gexy_dm, gexz_dm = get_dm_acc_amr(bbox=bbox, mass_dm=mass_dm)
    gexx = gexx_dm
    gexy = gexy_dm
    gexz = gexz_dm
    del gexx_dm, gexy_dm, gexz_dm
    
    # Get acceleration from stellar potential
    # mass_stellar = 1e9
    tmp_key = ('%-80s' % 'sim_mstars').encode('utf-8')
    mass_stellar = param_dict[tmp_key] / m_solar
    # scaling_r_stellar = 250.0
    tmp_key = ('%-80s' % 'sim_rstars').encode('utf-8')
    scaling_r_stellar = param_dict[tmp_key] / pc2cm
    # scaling_z_stellar = 100.0
    tmp_key = ('%-80s' % 'sim_zstars').encode('utf-8')
    scaling_z_stellar = param_dict[tmp_key] / pc2cm
    
    gexx_stellar, gexy_stellar, gexz_stellar = get_stellar_acc_amr(
        bbox=bbox, mass_stellar=mass_stellar,
        sc_rad=scaling_r_stellar, sc_z=scaling_z_stellar
    )
    gexx += gexx_stellar
    gexy += gexy_stellar
    gexz += gexz_stellar
    
    del gexx_stellar, gexy_stellar, gexz_stellar

# Get accelerations from rotation
(acx_rot, acy_rot, acz_rot), (velx, vely) = get_rotation_acc_amr(
    bbox=bbox,
    acx=gacx + gexx,
    acy=gacy + gexy,
    acz=gacz + gexz,
)

acc = [
    gacx + gexx + acx_rot,
    gacy + gexy + acy_rot,
    gacz + gexz + acz_rot,
]

# Set initial pressure at boundary edge
p0 = pres_init
start_bid = None

# Determine pressure from acceleration and density
pres_gc, bt = integrate_amr(
    dens=dens, acc=acc,
    bbox=bbox, gid=gid,
    blist=blist, p0=p0,
    start_bid=start_bid,
    debug=False
)

# Due to current limitations in the integration we have to
# mirror the data from the lower half to the upper half
# This should only be necessary when blocks go from a lower
# to a higher refinement level above the mid-plane.
# It requires block symmetry along that axis and is pretty
# time-consuming as we have to find corresponding mirror blocks
# one by one.
pres_gc[:, 1:9, 1:9, 1:9] = mirror_data(
    data=pres_gc[:, 1:9, 1:9, 1:9],
    coord=coord,
    blist=blist,
    ax=2
)

# Update all parents and get AMR data without GC
pres_gc[:, 1:9, 1:9, 1:9] = update_all_parent_blocks(
    data=pres_gc[:, 1:9, 1:9, 1:9], gid=gid
)
# Update all GCs
pres_gc = fill_ghostcells(
    data_pad=pres_gc, gid=gid, bbox=bbox, sel=None, debug=False
)

# Update rotation velocity due to acting radial pressure
vel_x, vel_y, acx_p, acy_p, acr_p = update_rotation_vel_amr(
    pres_gc=pres_gc, dens=dens,
    bbox=bbox, gid=gid,
    acx=gacx + gexx,
    acy=gacy + gexy,
    cut_reg=None
)





###############################################################################


axis = 2
pos = 0.0
ax_sl = [0, 1, 2]
ax_sl.pop(axis)

blist, brefs, bns = amr_tools.get_true_blocks(fname, xmin, xmax)

dens2 = dens[blist]
bbox2 = bbox[blist]
bs2 = pf['block size'][()][blist]
rlvl2 = pf['refine level'][()][blist]


pres = pres_gc[:, 1:9, 1:9, 1:9]
pres = (0.0*acx_p**2 + acy_p**2)**0.5

xmin, xmax = bbox[0, :, 0], bbox[-1, :, 1]
dx = (xmax - xmin) / (np.asarray(bns) * 2**(brefs[0] - brefs[1] + 3))
pres_im = amr_tools.get_slice(
    pres[blist], pos=1*dx[0], axis=axis, bbox=bbox[blist],
    ref_lvl=pf['refine level'][()][blist], bsize=pf['block size'][()][blist],
    bns=bns, brefs=brefs
)



sel = np.zeros(dens2.shape[0], dtype=int)
isIn = bbox2[:, axis, 0] < pos
isIn *= bbox2[:, axis, 1] >= pos
sel[isIn] = 1

recL = []
for i in range(sel.size):
    if sel[i] == 0:
        continue
    
    sq = bbox2[i, ax_sl, :] / pc2cm
    recL.append(
        Rectangle(
            (sq[0, 0], sq[1, 0]),
            sq[0, 1] - sq[0, 0],
            sq[1, 1] - sq[1, 0]
        )
    )
    

xmin = bbox2[0, :, 0] / pc2cm
xmax = bbox2[-1, :, 1] / pc2cm

pc = PatchCollection(recL, facecolor='none', edgecolor='k', lw=0.2, alpha=0.4)

fig = plt.figure()
ax = fig.gca()
ax.imshow(
    pres_im.T,
    extent=[xmin[0], xmax[0], xmin[1], xmax[1]],
    origin='lower', norm=LogNorm()
)
ax.add_collection(pc)
fig.show()
