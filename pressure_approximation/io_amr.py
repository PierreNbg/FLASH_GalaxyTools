import numpy as np
import flash_amr_tools as amr_tools
import h5py

# Reads the Fortran file and returns:
# ndim      - Number of dimensions of dataset
# shape     - Shape of dataset
# data      - Dataset in the order (nz, ny, nx) as applicable
def read_fortran(fname):
    # Open file as binary
    with open(fname, 'rb') as f:
        # First entry gives dimension of dataset
        # and is given in INT64
        ndim = np.fromfile(f, np.int64, 1)
        ndim = int(ndim)
        # Following ndim entries give the size in each dimension
        # and is given in INT64
        shape = np.fromfile(f, np.int64, ndim)
        # Read the remaining data entries (nx*ny*nz)
        # which are given in FLOAT64
        data = np.fromfile(f, np.float64, shape.prod())

    # Return dimension, shape and reshaped dataset in order
    # of z, y, x as applicable
    return ndim, shape, data.reshape(shape[::-1])


# Create dictionary from settings file
def read_settings(fname):
    # Read all lines of file
    with open(fname, 'r') as f:
        data = f.readlines()

    # Remove all empty lines and comments
    d_trim = [s for s in data if not (s.strip() == '' or '#' in s)]

    # Split entries into name and value (still strings)
    d_trim = [s.strip().split('=') for s in d_trim]

    # Create dictionary from key, value pairs
    # Evaluate value entries and convert if necessary
    di = dict([[s[0].strip(), eval(s[1].strip())] for s in d_trim])

    return di


def get_data_ext():
    # Read acceleration files
    # ndim and shape should be the same for all 3 arrays
    # Unit: [cm * s^-2]
    ndim, shape, acx = read_fortran('accx.out')
    ndim, shape, acy = read_fortran('accy.out')
    ndim, shape, acz = read_fortran('accz.out')
    
    print('acx, min, max', acx.min(), acx.max())
    print('acy, min, max', acy.min(), acy.max())
    print('acz, min, max', acz.min(), acz.max())
    
    # Transpose arrays to get x, y, z order from z, y, x order
    # Negate for grad P = - rho * a (not done here, chance for confusion)
    acx = -acx.T
    acy = -acy.T
    acz = -acz.T
    
    # Read acceleration files (DM)
    # ndim and shape should be the same for all 3 arrays
    # Unit: [cm * s^-2]
    # ndim, shape, acx_dm = read_fortran('accx_dm.out')
    # ndim, shape, acy_dm = read_fortran('accy_dm.out')
    # ndim, shape, acz_dm = read_fortran('accz_dm.out')
    
    # Read position arrays
    # These are only 1D, but we can get cell coordinates
    # from index of dataset with indices [j, k, i]
    # ==> x[i], y[j], z[k]
    # Unit: [cm]
    tmp, nx, x = read_fortran('x.out')
    tmp, ny, y = read_fortran('y.out')
    tmp, nz, z = read_fortran('z.out')
    
    # Cell size in each direction
    dx = (x[-1] - x[0]) / (nx - 1)
    dy = (y[-1] - y[0]) / (ny - 1)
    dz = (z[-1] - z[0]) / (nz - 1)
    print('dx', dx, dy, dz)
    
    # Read density array
    # Same shape as each of the acceleration arrays
    # Unit: [g cm^-3]
    ndim, shape, dens = read_fortran('density.out')
    print('density, min, max', dens.min(), dens.max())
    
    # Transpose arrays to get x, y, z order from z, y, x order
    dens = dens.T
    
    # Return main data
    # Accelerations from self-gravity [cm s^-2]
    # Positions (linear) [cm]
    # Size of x, y, z
    # Cell size [cm]
    # Density [g cm^-3]
    return (acx, acy, acz), (x, y, z), (nx, ny, nz), (dx, dy, dz), dens


def get_data_flash(fname, xmin=None, xmax=None):
    # Get all blocks within box xmin, xmax
    if xmax is None:
        xmax = []
    if xmin is None:
        xmin = []
    
    blist, brefs, bns = amr_tools.get_true_blocks(fname, xmin, xmax)
    
    blist = np.sort(blist)
    
    # Open HDF files
    pf = h5py.File(fname, 'r')
    keys = pf.keys()
    # Get gravitational acceleration in x, y and z
    # [cm s^-2]
    gacx = pf['gacx'][()][blist]
    gacy = pf['gacy'][()][blist]
    gacz = pf['gacz'][()][blist]
    
    ext_acc = False
    gexx, gexy, gexz = 0.0, 0.0, 0.0
    if 'gexx' in keys and 'gexx' in keys and 'gexx' in keys:
        gexx = pf['gexx'][()][blist]
        gexy = pf['gexy'][()][blist]
        gexz = pf['gexz'][()][blist]
        ext_acc = True
    
    # Get density [g cm^-3]
    dens = pf['dens'][()][blist]
    
    # Get bounding box of each block [cm]
    bbox = pf['bounding box'][()][blist]
    
    # Get block size of each block [cm]
    bsize = pf['block size'][()][blist]
    # Get refinement level of each block
    ref_lvl = pf['refine level'][()][blist]
    
    # Lower and upper corner of the region. Can be useful for other methods.
    low_cor, up_cor = (bbox[0, :, 0], bbox[-1, :, 1])
    
    # Put acceleration onto uniform grid
    acx = amr_tools.get_cube(
        data=gacx, bbox=bbox, bsize=bsize, ref_lvl=ref_lvl,
        brefs=brefs, bns=bns
    )
    acy = amr_tools.get_cube(
        data=gacy, bbox=bbox, bsize=bsize, ref_lvl=ref_lvl,
        brefs=brefs, bns=bns
    )
    acz = amr_tools.get_cube(
        data=gacz, bbox=bbox, bsize=bsize, ref_lvl=ref_lvl,
        brefs=brefs, bns=bns
    )
    
    # Get acceleration from external potentials
    acx_ex, acy_ex, acz_ex = 0.0, 0.0, 0.0
    if ext_acc:
        acx_ex = amr_tools.get_cube(
            data=gexx, bbox=bbox, bsize=bsize, ref_lvl=ref_lvl,
            brefs=brefs, bns=bns
        )
        acy_ex = amr_tools.get_cube(
            data=gexy, bbox=bbox, bsize=bsize, ref_lvl=ref_lvl,
            brefs=brefs, bns=bns
        )
        acz_ex = amr_tools.get_cube(
            data=gexz, bbox=bbox, bsize=bsize, ref_lvl=ref_lvl,
            brefs=brefs, bns=bns
        )
        
    # Read density array
    # Same shape as each of the acceleration arrays
    # Unit: [g cm^-3]
    dens = amr_tools.get_cube(
        data=dens, bbox=bbox, bsize=bsize, ref_lvl=ref_lvl,
        brefs=brefs, bns=bns
    )
    print('density, min, max', dens.min(), dens.max())
    
    nx, ny, nz = dens.shape
    
    # Determine cell size
    dx, dy, dz = (up_cor - low_cor) / dens.shape
    
    x = np.linspace(low_cor[0], up_cor[0], nx, endpoint=False) + dx / 2.
    y = np.linspace(low_cor[1], up_cor[1], ny, endpoint=False) + dy / 2.
    z = np.linspace(low_cor[2], up_cor[2], nz, endpoint=False) + dz / 2.
    
    print('X (min, max):', x[0], x[-1])
    print('Y (min, max):', y[0], y[-1])
    print('Z (min, max):', z[0], z[-1])
    
    print('acx, min, max', acx.min(), acx.max())
    print('acy, min, max', acy.min(), acy.max())
    print('acz, min, max', acz.min(), acz.max())
    
    # Return main data
    # Accelerations from self-gravity [cm s^-2]
    # Positions (linear) [cm]
    # Size of x, y, z
    # Cell size [cm]
    # Density [g cm^-3]
    return (acx, acy, acz), (acx_ex, acy_ex, acz_ex), (x, y, z), (nx, ny, nz), (dx, dy, dz), dens


# Update rotation velocity with impact from pressure curves
# which partially stabilizes the disk in radial direction
def update_rotation_vel(pres, dens, x, y, z, acx, acy, cut_reg=None):
    # Update pressure to be uniform in mid-plane
    # p_cor = pres.max() - pres[:,:,pres.shape[2]//2]
    
    # pnew = pres + p_cor[:,:,None]
    
    # Get cell positions
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    # Get cylindrical radius
    rr = (xx ** 2 + yy ** 2) ** 0.5
    
    # Initialize fields to 0
    # of grad P (Hydrostatic equilibrium)
    pgradx = np.zeros_like(pres)
    pgrady = np.zeros_like(pres)
    
    # Determine central difference for x and y component
    pgradx[1:-1, :, :] = (pres[2:, :, :] - pres[:-2, :, :]) / (xx[2:, :, :] - xx[:-2, :, :])
    pgradx[0, :, :] = (pres[1, :, :] - pres[0, :, :]) / (xx[1, :, :] - xx[0, :, :])
    pgradx[-1, :, :] = (pres[-2, :, :] - pres[-1, :, :]) / (xx[-2, :, :] - xx[-1, :, :])
    
    pgrady[:, 1:-1, :] = (pres[:, 2:, :] - pres[:, :-2, :]) / (yy[:, 2:, :] - yy[:, :-2, :])
    pgrady[:, 0, :] = (pres[:, 1, :] - pres[:, 0, :]) / (yy[:, 1, :] - yy[:, 0, :])
    pgrady[:, -1, :] = (pres[:, -2, :] - pres[:, -1, :]) / (yy[:, -2, :] - yy[:, -1, :])
    
    # Use hydrostatic equilibrium to get acceleration due to pressure profile
    acx_p = -pgradx / dens
    acy_p = -pgrady / dens
    
    # Get angle of radial vector
    phi = np.arctan2(yy, xx)
    
    # Get radial component of total acceleration (gas+dm+pres)
    # ac* = (gas + dm), ac*_p = (pressure)
    # acr = (acx_p + acx) * np.cos(phi) + (acy_p + acy) * np.sin(phi)
    acr = (acx - acx_p) * np.cos(phi) + (acy - acy_p) * np.sin(phi)
    
    # Get absolute rotation velocity
    # from dark matter acceleration
    # v = (r * d/dr phi)**0.5 = (-r * acc)**0.5
    v_abs = (acr * rr) ** 0.5
    # v_abs = np.nan_to_num(v_abs)
    
    # Determine x and y components of rotation
    vel_x = -v_abs * np.sin(phi)
    vel_y = v_abs * np.cos(phi)
    
    # Remove nan from rr = 0
    vel_x = np.nan_to_num(vel_x)
    vel_y = np.nan_to_num(vel_y)
    
    if cut_reg is not None:
        vel_x = np.where(cut_reg, vel_x, 0)
        vel_y = np.where(cut_reg, vel_y, 0)
        acx_p = np.where(cut_reg, acx_p, 0)
        acy_p = np.where(cut_reg, acy_p, 0)
    
    # velx_cor = vx.copy()
    # vely_cor = vy.copy()
    
    # velx_cor[1:-1, 1:-1, :] += vel_x[1:-1, 1:-1, :]
    # vely_cor[1:-1, 1:-1, :] += vel_y[1:-1, 1:-1, :]
    
    # velx_cor += vel_x
    # vely_cor += vel_y
    
    return vel_x, vel_y, acx_p, acy_p


# Determine the new temperatures from the updated pressures:
# pres_old / (rho_old * temp_old) = const. = pres_new / (rho_new * temp_new)
# temp_new = (pres_new / pres_old) * (rho_old / rho_new) * temp_old
#   rho_old == rho_new
# temp_new = (pres_new / pres_old) * temp_old
def get_new_temp(temp_old, pres_old, pres_new):
    print('\tUpdating temperature')
    print('\tOLD (min, max): %1.3e\t%1.3e' % (temp_old.min(), temp_old.max()))
    ratio = pres_new / pres_old
    temp_new = temp_old * ratio
    print('\tNEW (min, max): %1.3e\t%1.3e' % (temp_new.min(), temp_new.max()))
    return temp_new


# Determine the new internal energies from the updated pressures:
# eint_old * rho_old / pres_old = const. = eint_new * rho_new / pres_new
# eint_new = (pres_new / pres_old) * (rho_old / rho_new) * eint_old
#   rho_old == rho_new
# eint_new = (pres_new / pres_old) * eint_old
def get_new_eint(eint_old, pres_old, pres_new):
    print('\tUpdating internal energy')
    print('\tOLD (min, max): %1.3e\t%1.3e' % (eint_old.min(), eint_old.max()))
    ratio = pres_new / pres_old
    eint_new = eint_old * ratio
    print('\tNEW (min, max): %1.3e\t%1.3e' % (eint_new.min(), eint_new.max()))
    return eint_new


# Update total energy with updated new internal energies
def get_new_ener(ener_old, eint_old, eint_new, velx_new=None, vely_new=None):
    print('\tUpdating total energy')
    print('\tOLD (min, max): %1.3e\t%1.3e' % (ener_old.min(), ener_old.max()))
    if type(velx_new) == type(None) and type(vely_new) == type(None):
        ener_new = ener_old - eint_old
    else:
        ener_new = 0.5 * (velx_new ** 2 + vely_new ** 2)
    
    ener_new += eint_new
    print('\tNEW (min, max):  %1.3e\t%1.3e' % (ener_new.min(), ener_new.max()))
    return ener_new


# Create a copy of the old checkpoint file and update the fields
# pres, temp, eint and ener with new values.
def create_new_hdf(pf, fname, pres, temp, eint, ener, velx, vely):
    if pf.filename == fname:
        print('Old and new file have same file name.\nABORT to prevent overriding data.')
        return
    
    # Get key list of the old file
    keys_old = list(pf.keys())
    
    # Remove keys we want to update from key list
    k_pres = keys_old.pop(keys_old.index('pres'))
    k_temp = keys_old.pop(keys_old.index('temp'))
    k_eint = keys_old.pop(keys_old.index('eint'))
    k_ener = keys_old.pop(keys_old.index('ener'))
    k_velx = keys_old.pop(keys_old.index('velx'))
    k_vely = keys_old.pop(keys_old.index('vely'))
    
    # Open new file
    pf_new = h5py.File(fname, 'w')
    
    # Copy all data which is not modified from the old file to the new file
    for k in keys_old:
        print('Add field', k)
        pf_new.create_dataset('/' + k, data=pf[k])
    
    pf_new.create_dataset('/pres', data=pres)
    pf_new.create_dataset('/temp', data=temp)
    pf_new.create_dataset('/eint', data=eint)
    pf_new.create_dataset('/ener', data=ener)
    pf_new.create_dataset('/velx', data=velx)
    pf_new.create_dataset('/vely', data=vely)
    
    pf_new.close()


# Put the data of a uniform grid back onto the AMR data structure of FLASH.
# Lower refined blocks are determined by averaging.
def put_ugrid_onto_blocklist(pf, new_dat, x, y, z):
    # Get the bounding boxes of the blocks
    bbox = pf['bounding box'][()]
    # Create new array with the shape of the AMR structure
    dat_field = np.zeros_like(pf['dens'][()])
    
    # Get the 1D x, y and z positions of the uniform grid
    xx = np.unique(x)
    yy = np.unique(y)
    zz = np.unique(z)
    
    pro = list(np.arange(11) * 10.)
    
    # Loop over all blocks with the AMR structure
    for i in range(bbox.shape[0]):
        if 100. * i / bbox.shape[0] >= pro[0]:
            print(np.round(100. * i / bbox.shape[0]), '%')
            pro.pop(0)
        # Find the lower and upper index of the uniform grid which corresponds to the current block
        # in x, y and z-direction
        xl, xu = np.where((xx >= bbox[i, 0, 0]) * (xx <= bbox[i, 0, 1]))[0][[0, -1]]
        yl, yu = np.where((yy >= bbox[i, 1, 0]) * (yy <= bbox[i, 1, 1]))[0][[0, -1]]
        zl, zu = np.where((zz >= bbox[i, 2, 0]) * (zz <= bbox[i, 2, 1]))[0][[0, -1]]
        
        # Create slices which correspond to these cells
        slx = slice(xl, xu + 1, None)
        sly = slice(yl, yu + 1, None)
        slz = slice(zl, zu + 1, None)
        
        # Select data (3D)
        # sh_new = (np.unique(xs).size, np.unique(ys).size, np.unique(zs).size)
        dat = new_dat[slx, sly, slz].copy()
        # dat = new_dat[tr_sl].reshape(sh_new)
        # dat = new_dat[xs, ys, zs]
        # print(dat.shape, pf['gid'][()][i, -1] == -1)
        
        # Get the number of cells within the region
        # Should be always in a shape of (nx*8, ny*8, nz*8)
        # Where nx, ny and nz represent the number of blocks in each
        # direction. nx, ny and nz should be the same value
        nr_cells = dat.size
        
        # If the number of cells corresponds to a multiple of 512 (8**3)
        # We put the data onto the AMR structure
        if nr_cells % 512 == 0:
            # If we are at a lower refined block we need to average neighbouring cells
            # We need to determine the number of levels we have to "derefine" our data
            lvl = int(np.log2(nr_cells // 512) // 3)
            # Loop over all levels
            for j in range(lvl):
                # lvl -= 1
                dat_tmp = dat[0::2, 0::2, 0::2]  # 0, 0, 0
                dat_tmp += dat[0::2, 0::2, 1::2]  # 0, 0, 1
                dat_tmp += dat[0::2, 1::2, 0::2]  # 0, 1, 0
                dat_tmp += dat[0::2, 1::2, 1::2]  # 0, 1, 1
                dat_tmp += dat[1::2, 0::2, 0::2]  # 1, 0, 0
                dat_tmp += dat[1::2, 0::2, 1::2]  # 1, 0, 1
                dat_tmp += dat[1::2, 1::2, 0::2]  # 1, 1, 0
                dat_tmp += dat[1::2, 1::2, 1::2]  # 1, 1, 1
                dat_tmp /= 8.
                dat = dat_tmp
            
            # Store the data in the AMR structure
            dat_field[i, :, :, :] = dat.T
        
        else:
            print('ERROR: Selection found odd number of cells')
            print('Nr. cells: ', nr_cells, ' at block ID: ', i)
            break
    
    return dat_field

# Update all parents with data from their children
def update_all_parent_blocks(data, gid, sel=None):
    # Get list of block ids
    # Going backwards through the list
    # will automatically fill all parents
    blist = np.arange(gid.shape[0])[::-1]
    
    if sel is not None:
        blist = sel
        if type(blist) is int:
            blist = [blist]
        blist = blist[::-1]
    
    # List of children ids
    cid_list = np.arange(8)
    
    # Loop over all blocks
    for bid in blist:
        # Get block info
        blk_info = gid[bid] - 1
        # Get block index of parent
        bid_par = blk_info[6]
        
        # If parent does not exist go to next block
        if bid_par == -1 - 1:
            continue
        
        # Get children of parent
        bid_child_par = gid[bid_par, 7:] - 1
        # Find child id of current block in parent
        cid = cid_list[bid_child_par == bid][0]
        
        # Determine position of child in parent
        shift_xyz = [
            (cid // 4) % 2,
            (cid // 2) % 2,
            cid % 2,
        ]
        
        # Create slices the place to put our child data
        sl_z = slice(4 * shift_xyz[0], 4 * (shift_xyz[0] + 1))
        sl_y = slice(4 * shift_xyz[1], 4 * (shift_xyz[1] + 1))
        sl_x = slice(4 * shift_xyz[2], 4 * (shift_xyz[2] + 1))
        
        # Downsample data into parent
        # Make sure it is 0.0 before we add data
        data[bid_par][sl_z, sl_y, sl_x] = 0.0
        data[bid_par][sl_z, sl_y, sl_x] += data[bid][0::2, 0::2, 0::2]  # 0 0 0
        data[bid_par][sl_z, sl_y, sl_x] += data[bid][0::2, 0::2, 1::2]  # 0 0 1
        data[bid_par][sl_z, sl_y, sl_x] += data[bid][0::2, 1::2, 0::2]  # 0 1 0
        data[bid_par][sl_z, sl_y, sl_x] += data[bid][0::2, 1::2, 1::2]  # 0 1 1
        data[bid_par][sl_z, sl_y, sl_x] += data[bid][1::2, 0::2, 0::2]  # 1 0 0
        data[bid_par][sl_z, sl_y, sl_x] += data[bid][1::2, 0::2, 1::2]  # 1 0 1
        data[bid_par][sl_z, sl_y, sl_x] += data[bid][1::2, 1::2, 0::2]  # 1 1 0
        data[bid_par][sl_z, sl_y, sl_x] += data[bid][1::2, 1::2, 1::2]  # 1 1 1
        data[bid_par][sl_z, sl_y, sl_x] /= 8.0
    
    return data

def mirror_data(data, coord, blist, ax=2):
    # Get total number of block that exist in the domain
    nrblks = coord.shape[0]
    
    # Create default slice set which mirrors along given axis
    sl = [
        slice(None, None, None),
        slice(None, None, None),
        slice(None, None, None)
    ]
    
    # Select axis which is mirrored
    sl[2 - ax] = slice(None, None, -1)
    
    # Create tracker of blocks that have been touched
    track_blocks = np.zeros(nrblks)
    # Loop over all blocks in blist
    for i in blist:
        # If block has been updated already
        # go to next block
        if track_blocks[i]:
            continue
        
        # Get coordinates of current block
        coord_cblk = coord[i].copy()
        # Flip invert coordinate of given axis
        coord_cblk[ax] = -coord_cblk[ax]
        
        # Find block which lies opposite
        sel_mirror = (np.abs(coord - coord_cblk) < 1e-2).prod(axis=1).astype(bool)
        
        # If block is found update data
        if sel_mirror.sum() > 0:
            track_blocks[i] = True
            track_blocks[sel_mirror.tolist()] = True
            
            data[sel_mirror.tolist()] = data[i, *sl]
    # Return new dataset
    return data


