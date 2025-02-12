import sys

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
    update_all_parent_blocks, mirror_data,
)

from constants import m_solar, pc2cm

def main(fname, xmin, xmax, pres_init):
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
        #mass_dm = 1e12
        mass_dm = param_dict[('%-80s' % 'sim_m0').encode('utf-8')] / m_solar
        
        gexx_dm, gexy_dm, gexz_dm = get_dm_acc_amr(bbox=bbox, mass_dm=mass_dm)
        gexx = gexx_dm
        gexy = gexy_dm
        gexz = gexz_dm
        del gexx_dm, gexy_dm, gexz_dm

        # Get acceleration from stellar potential
        #mass_stellar = 1e9
        mass_stellar = param_dict[('%-80s' % 'sim_mstars').encode('utf-8')] / m_solar
        #scaling_r_stellar = 250.0
        scaling_r_stellar = param_dict[('%-80s' % 'sim_rstars').encode('utf-8')] / pc2cm
        #scaling_z_stellar = 100.0
        scaling_z_stellar = param_dict[('%-80s' % 'sim_zstars').encode('utf-8')] / pc2cm
        
        gexx_stellar, gexy_stellar, gexz_stellar = get_stellar_acc_amr(
            bbox=bbox, mass_stellar=mass_stellar,
            sc_rad=scaling_r_stellar, sc_z=scaling_z_stellar
        )
        gexx += gexx_stellar
        gexy += gexy_stellar
        gexz += gexz_stellar
        
        del gexx_stellar, gexy_stellar, gexz_stellar

    # Get accelerations from rotation
    (acx_rot, acy_rot, acz_rot), (velx, vely)= get_rotation_acc_amr(
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
    #p0 = 3e-8
    #start_bid = 38487
    #start_bid = 21993
    #p0 = 5e-9
    #start_bid = 31493

    #p0 = 1e-14
    p0 = pres_init
    start_bid = None
    
    # Determine pressure from acceleration and density
    pres_gc = integrate_amr(dens=dens, acc=acc, bbox=bbox, gid=gid, blist=blist, p0=p0, start_bid=start_bid, debug=False)
    
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
    pres_gc[:, 1:9, 1:9, 1:9] = update_all_parent_blocks(data=pres_gc[:, 1:9, 1:9, 1:9], gid=gid)
    # Update all GCs
    pres_gc = fill_ghostcells(
        data_pad=pres_gc, gid=gid, bbox=bbox, sel=None, debug=False
    )
    
    # Update rotation velocity due to acting radial pressure
    vel_x, vel_y, acx_p, acy_p = update_rotation_vel_amr(
        pres_gc=pres_gc, dens=dens,
        bbox=bbox, gid=gid,
        acx=gacx + gexx,
        acy=gacy + gexy,
        cut_reg=None
    )

    # Update temperature, internal energy and total energy with new pressures
    temp = get_new_temp(temp_old=pf['temp'][()], pres_old=pf['pres'][()], pres_new=pres_gc[:, 1:9, 1:9, 1:9])
    eint = get_new_eint(eint_old=pf['eint'][()], pres_old=pf['pres'][()], pres_new=pres_gc[:, 1:9, 1:9, 1:9])
    ener = get_new_ener(
        ener_old=pf['ener'][()], eint_old=pf['eint'][()],
        eint_new=eint, velx_new=vel_x, vely_new=vel_y
    )
    
    # Create new checkpoint file with updated quantities
    create_new_hdf(
        pf=pf, fname=fname + '_HSE',
        pres=pres_gc[:, 1:9, 1:9, 1:9], temp=temp,
        eint=eint, ener=ener,
        velx=velx, vely=vely
    )

fname = sys.argv[1]
pres_init = float(sys.argv[2])
xmin = []
xmax = []
main(fname, xmin, xmax, pres_init)
