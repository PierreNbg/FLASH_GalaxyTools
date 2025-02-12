from __future__ import print_function

#from pytreegrav import Accel
import numpy as np
import h5py
import warnings
warnings.simplefilter('ignore', np.RankWarning)

from io_amr import (
    get_data_flash,
    update_rotation_vel,
    put_ugrid_onto_blocklist,
    create_new_hdf,
    get_new_eint,
    get_new_ener,
    get_new_temp
)
from external_potentials import (
    get_g_dm,
    get_g_stellar,
    get_dm_rot
)
from pressure_determination import get_pressures
#from constants import *


def main_chk():
    fname_chk = 'DwarfGal_hdf5_chk_0000'

    xmin = []
    xmax = []

    acc, acc_ex, pos, sh, dx, dens = get_data_flash(fname=fname_chk, xmin=xmin, xmax=xmax)

    # Separate data products into x-, y- and z-components
    acx, acy, acz = acc
    acx_ex, acy_ex, acz_ex = acc_ex
    x, y, z = pos
    nx, ny, nz = sh
    dx, dy, dz = dx
    
    determine_external = False
    if type(acx_ex) is float:
        determine_external = True

    # Pnt acceleration due to self gravity
    print('acx', np.abs(acx).min(), np.abs(acx).max())
    print('acy', np.abs(acy).min(), np.abs(acy).max())
    print('acz', np.abs(acz).min(), np.abs(acz).max())

    # xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

    if determine_external:
        # Determine acceleration due to dark matter
        acx_dm, acy_dm, acz_dm = get_g_dm(x=x, y=y, z=z)
        print('acx_dm', np.abs(acx_dm).min(), np.abs(acx_dm).max())
        print('acy_dm', np.abs(acy_dm).min(), np.abs(acy_dm).max())
        print('acz_dm', np.abs(acz_dm).min(), np.abs(acz_dm).max())
    
        acx_star, acy_star, acz_star = get_g_stellar(x=x, y=y, z=z)
        print('acx_star', np.abs(acx_star).min(), np.abs(acx_star).max())
        print('acy_star', np.abs(acy_star).min(), np.abs(acy_star).max())
        print('acz_star', np.abs(acz_star).min(), np.abs(acz_star).max())
        print(acx_star[0,0,0], acy_star[0,0,0], acx_star[0,0,0])
    
        # Determine acceleration due to rotation of disk
        (acx_rot, acy_rot, acz_rot), (velx, vely) = get_dm_rot(
            x=x, y=y, z=z,
            acx_dm=acx_dm+acx+acx_star,
            acy_dm=acy_dm+acy+acy_star,
            acz_dm=acz_dm+acz+acz_star,
        )
        print('acx_rot', np.abs(acx_rot).min(), np.abs(acx_rot).max())
        print('acy_rot', np.abs(acy_rot).min(), np.abs(acy_rot).max())
        print('acz_rot', np.abs(acz_rot).min(), np.abs(acz_rot).max())
    
        # Combine all accelerations
        # Add to acx, acy, acz
        acx += acx_dm + acx_rot + acx_star
        acy += acy_dm + acy_rot + acy_star
        acz += acz_dm + acz_rot + acz_star
        
        del acx_dm, acy_dm, acz_dm
        del acx_star, acy_star, acz_star
    else:
        # Determine acceleration due to rotation of disk
        (acx_rot, acy_rot, acz_rot), (velx, vely) = get_dm_rot(
            x=x, y=y, z=z,
            acx_dm=acx + acx_ex,
            acy_dm=acy + acy_ex,
            acz_dm=acz + acz_ex,
        )
        
        # Combine all accelerations
        # Add to acx, acy, acz
        acx += acx_ex + acx_rot
        acy += acy_ex + acy_rot
        acz += acz_ex + acz_rot
        
        del acx_ex, acy_ex, acz_ex

    # Get pressures using the ordered integrations scheme
    pres_zrr, pres_rzr, pres_rrz = get_pressures(
        acx=acx, acy=acy, acz=acz, dens=dens,
        dx=dx, dy=dy, dz=dz
    )

    velx_cor, vely_cor, acx_pres, acy_pres = update_rotation_vel(
        pres=pres_rrz, dens=dens,
        x=x, y=y, z=z,
        acx=acx_rot, acy=acy_rot,
        #cut_reg=dens > 1e-28
    )
    
    # Open old checkpoint file
    pf = h5py.File(fname_chk, 'r')
    # Put uniform grid data back in AMR structure
    pres_new_field = put_ugrid_onto_blocklist(
        pf=pf, new_dat=pres_rrz,
        x=x, y=y, z=z
    )
    velx_new_field = put_ugrid_onto_blocklist(
        pf=pf, new_dat=velx_cor,
        x=x, y=y, z=z
    )
    vely_new_field = put_ugrid_onto_blocklist(
        pf=pf, new_dat=vely_cor,
        x=x, y=y, z=z
    )
    # Get new temperature
    temp_new_field = get_new_temp(
        temp_old=pf['temp'][()], pres_old=pf['pres'][()],
        pres_new=pres_new_field
    )
    # Get new internal energy
    eint_new_field = get_new_eint(
        eint_old=pf['eint'][()], pres_old=pf['pres'][()],
        pres_new=pres_new_field
    )
    # Get new total energy
    ener_new_field = get_new_ener(
        ener_old=pf['ener'][()], eint_old=pf['eint'][()],
        eint_new=eint_new_field,
        velx_new=velx_new_field, vely_new=vely_new_field
    )
    # Create new file
    create_new_hdf(
        pf=pf, fname=pf.filename + '_HSE',
        pres=pres_new_field, temp=temp_new_field,
        eint=eint_new_field, ener=ener_new_field,
        velx=velx_new_field, vely=vely_new_field

    )

    return


main_chk()


