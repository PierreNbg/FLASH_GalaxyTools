import numpy as np

from constants import m_solar, pc2cm, kpc2cm, g_newt
from integration_amr import get_cell_centers, fill_ghostcells, create_ghostcells


def get_dm_acc_amr(bbox, mass_dm):
    ccoords = get_cell_centers(bbox)
    
    # Separate components
    xx = ccoords[..., 0]
    yy = ccoords[..., 1]
    zz = ccoords[..., 2]
    
    # Determine distance to center of galaxy
    rr = ((ccoords / pc2cm)**2).sum(axis=-1)**0.5
    rr *= pc2cm
    
    # Dark matter mass [g]
    m0 = mass_dm * m_solar
    print('Dark matter potential from Burkert:')
    print('\tDark matter mass (Msun):\t%1.3e' % mass_dm)
    
    # r0 in cm
    r0 = 3.07 * kpc2cm * (m0 / (m_solar * 1e9)) ** (3. / 7.)
    # rhod0 in g cm^-3
    rhod0 = 1.46e-24 * (m0 / (m_solar * 1e9)) ** (-2 / 7.)
    konst = np.pi * g_newt * rhod0 * r0 ** 2
    
    # Calculate the absolute acceleration
    gacc = -(rr**2 + r0 * rr) / (r0**2 + rr**2)
    gacc += np.arctan(rr / r0)
    gacc += (rr**2 + r0 * rr) / (r0**2 + rr * r0)
    gacc -= np.log(1. + rr / r0)
    gacc -= (rr**3 - r0 * rr**2) / (r0**3 + r0 * rr**2)
    gacc -= 0.5 * np.log(1. + (rr / r0)**2)
    gacc *= -2.0 * konst * r0 / rr**3
    
    # Remove nan that can occur for rr = 0
    gacc = np.nan_to_num(gacc)
    
    # Get acceleration for each component
    acx_dm = -gacc * xx
    acy_dm = -gacc * yy
    acz_dm = -gacc * zz
    
    print('\tacceleration in x (min, max):', np.abs(acx_dm).min(), np.abs(acx_dm).max())
    print('\tacceleration in y (min, max):', np.abs(acy_dm).min(), np.abs(acy_dm).max())
    print('\tacceleration in z (min, max):', np.abs(acz_dm).min(), np.abs(acz_dm).max())
    
    return acx_dm, acy_dm, acz_dm


# Old stellar population Miyamoto & Nagai
def get_stellar_acc_amr(bbox, mass_stellar, sc_rad, sc_z):
    ccoords = get_cell_centers(bbox)
    
    # Separate components
    xx = ccoords[..., 0]
    yy = ccoords[..., 1]
    zz = ccoords[..., 2]
    
    # Determine distance to center of galaxy
    rr = (xx ** 2 + yy ** 2)**0.5
    
    # Mass of stellar population
    mass = mass_stellar * m_solar
    # Scaling in radius
    a = sc_rad * pc2cm
    # Scaling in height
    b = sc_z * pc2cm
    
    print('Stellar potential from Miyamoto & Nagai:')
    
    print('\tStellar mass (Msun):\t%1.3e' % mass_stellar)
    print('\tRadial scale (pc):\t', sc_rad)
    print('\tHeight scale (pc):\t', sc_z)
    
    # Pre-factor
    konst = g_newt * mass
    
    # Common component of acceleration
    g = konst * (rr**2 + (a + (zz**2 + b**2)**0.5)**2)**(-1.5)
    
    # Get acceleration for each component
    accx_star = -g * xx
    accy_star = -g * yy
    accz_star = -g * zz * (1. + a / (zz**2 + b**2)**0.5)
    
    print('\tacceleration in x (min, max):', np.abs(accx_star).min(), np.abs(accx_star).max())
    print('\tacceleration in y (min, max):', np.abs(accy_star).min(), np.abs(accy_star).max())
    print('\tacceleration in z (min, max):', np.abs(accz_star).min(), np.abs(accz_star).max())

    
    return accx_star, accy_star, accz_star


# Get rotation from given acceleration
# Assumes ration around z-axis
# Only radial (cylindrical) acceleration of x- and y-component are considered
def get_rotation_acc_amr(bbox, acx, acy, acz, return_vel=False, cut_reg=None):
    ccoords = get_cell_centers(bbox)
    
    # Separate components
    xx = ccoords[..., 0]
    yy = ccoords[..., 1]
    # zz = ccoords[..., 2]
    
    # Determine cylindrical distance to center
    rr = (xx ** 2 + yy ** 2) ** 0.5
    
    # Azimuthal angle
    phi = np.arctan2(yy, xx)
    # Radial acceleration
    acr = acx * np.cos(phi) + acy * np.sin(phi)
 
    # Counter-acting acceleration from centrifugal force
    a_abs = -acr
    
    # Get absolute rotation velocity
    # from dark matter acceleration:
    #   v = (r * d/dr phi)**0.5
    #     = (-r * acc)**0.5
    v_abs = (a_abs * rr) ** 0.5
    
    print('Rotation from acting accelerations')
    print('\tVelocity range (km/s) (min/max):\t', v_abs.min() / 1e5, v_abs.max() / 1e5)

    if return_vel:
        return rr, v_abs

    # Determine x and y components of rotation.
    # Rotation is counter-clockwise
    vel_x = -v_abs * np.sin(phi)
    vel_y = v_abs * np.cos(phi)
    
    # Remove nan from rr = 0
    vel_x = np.nan_to_num(vel_x)
    vel_y = np.nan_to_num(vel_y)
    
    # Absolute acceleration from centripetal force
    acc_rot = (vel_x ** 2 + vel_y ** 2) / rr
    # Remove nan from r = 0
    acc_rot = np.nan_to_num(acc_rot)
    
    # Determine acceleration for x and y component
    acx_rot = acc_rot * np.cos(phi)
    acy_rot = acc_rot * np.sin(phi)
    
    acx_rot = -acx
    acy_rot = -acy
    
    # Remove nan from r = 0
    acx_rot = np.nan_to_num(acx_rot)
    acy_rot = np.nan_to_num(acy_rot)
    # z component always 0 (rotation axis)
    acz_rot = np.zeros_like(acx_rot)
    
    # Allow for disabling rotation under given criteria
    if cut_reg is not None:
        vel_x = np.where(cut_reg, vel_x, 0)
        vel_y = np.where(cut_reg, vel_y, 0)
        acx_rot = np.where(cut_reg, acx_rot, 0)
        acy_rot = np.where(cut_reg, acy_rot, 0)
    
    return (acx_rot, acy_rot, acz_rot), (vel_x, vel_y)


# Update rotation velocity with impact from pressure curves
# which partially stabilizes the disk in radial direction
def update_rotation_vel_amr(pres_gc, dens, bbox, gid, acx, acy, cut_reg=None):
    # Get cell centers of each cell
    ccoords = get_cell_centers(bbox)
    
    gc_slice = [
        slice(1, 9, None),
        slice(1, 9, None),
        slice(1, 9, None)
    ]
    
    # Separate components
    xx = ccoords[..., 0]
    yy = ccoords[..., 1]
    zz = ccoords[..., 2]
    
    #blist = np.arange(pres_gc.shape[0])
    #blist = blist[gid[:, -1] == -1].tolist()
    
    xx_gc = create_ghostcells(data=xx)
    xx_gc = fill_ghostcells(data_pad=xx_gc, gid=gid, bbox=bbox)
    yy_gc = create_ghostcells(data=yy)
    yy_gc = fill_ghostcells(data_pad=yy_gc, gid=gid, bbox=bbox)

    # Get cylindrical radius
    rr = (xx ** 2 + yy ** 2) ** 0.5
    
    # Get cell size in each block
    #dx = (bbox[:, :, 1] - bbox[:, :, 0]) / dens.shape[1:]
    
    #sl_xx_prev = gc_slice.copy()
    #sl_xx_next = gc_slice.copy()
    #sl_xx_prev[2 - 0] = 0
    #sl_xx_next[2 - 0] = 1
    #xx_gc[:, *sl_xx_prev] = xx_gc[:, *sl_xx_next] - dx[:, 0][:, None, None]
    #sl_xx_prev[2 - 0] = 9
    #sl_xx_next[2 - 0] = 8
    #xx_gc[:, *sl_xx_prev] = xx_gc[:, *sl_xx_next] + dx[:, 0][:, None, None]

    #sl_yy_prev = gc_slice.copy()
    #sl_yy_next = gc_slice.copy()
    #sl_yy_prev[2 - 1] = 0
    #sl_yy_next[2 - 1] = 1
    #yy_gc[:, *sl_yy_prev] = yy_gc[:, *sl_yy_next] - dx[:, 1][:, None, None]
    #sl_yy_prev[2 - 1] = 9
    #sl_yy_next[2 - 1] = 8
    #yy_gc[:, *sl_yy_prev] = yy_gc[:, *sl_yy_next] + dx[:, 1][:, None, None]

    # pgradx = determine_grad(pres_gc=pres_gc, gid=gid, dx=dx, axis=0, use_gc_for_no_neigh=True)
    pgradx = determine_grad(pres_gc=pres_gc, gid=gid, pos=xx_gc, axis=0, use_gc_for_no_neigh=True)
    # pgrady = determine_grad(pres_gc=pres_gc, gid=gid, dx=dx, axis=1, use_gc_for_no_neigh=True)
    pgrady = determine_grad(pres_gc=pres_gc, gid=gid, pos=yy_gc, axis=1, use_gc_for_no_neigh=True)
    
    # Use hydrostatic equilibrium to get acceleration due to pressure profile
    acx_p = -pgradx / dens
    acy_p = -pgrady / dens

    # Get angle of radial vector
    phi = np.arctan2(yy, xx)
    
    acr_p = acx_p * np.cos(phi) + acy_p * np.sin(phi)
    
    # Get radial component of total acceleration (gas+ext+pres)
    # ac* = (gas + ext), ac*_p = (pressure)
    # acr = (acx_p + acx) * np.cos(phi) + (acy_p + acy) * np.sin(phi)
    acr = (acx + acx_p) * np.cos(phi) + (acy + acy_p) * np.sin(phi)
    # Flip sign as we want to derive the velocity necessary to counter-act
    # the inwards acceleration
    acr = -acr
    # Get absolute rotation velocity
    # from dark matter acceleration
    # v = (r * d/dr phi)**0.5 = (-r * acr)**0.5
    # Any point which is already counter-acted by pressure
    # will be set 0.0 velocity
    v_abs = (acr * rr) ** 0.5
    v_abs = np.nan_to_num(v_abs)
    
    print('Rotation after correction for radial pressure gradient')
    print('\tVelocity range (km/s) (min/max):\t', v_abs.min() / 1e5, v_abs.max() / 1e5)
    
    # Determine x and y components of rotation
    # Counter-clockwise rotation
    vel_x = -v_abs * np.sin(phi)
    vel_y = v_abs * np.cos(phi)
    
    # Remove nan from rr = 0
    vel_x = np.nan_to_num(vel_x)
    vel_y = np.nan_to_num(vel_y)
    
    # Apply a cutting region at which the values are set to 0.0
    if cut_reg is not None:
        vel_x = np.where(cut_reg, vel_x, 0)
        vel_y = np.where(cut_reg, vel_y, 0)
        acx_p = np.where(cut_reg, acx_p, 0)
        acy_p = np.where(cut_reg, acy_p, 0)
    
    return vel_x, vel_y, acx_p, acy_p, acr_p


# Check outlier in dataset and smooth them by taking the average of neighbours
def smooth_data(acr, gid, bbox, del_acr=0.05):
    acr_gc = create_ghostcells(data=acr)
    acr_gc = fill_ghostcells(data_pad=acr_gc, gid=gid, bbox=bbox, sel=None)
    
    # Smooth in x
    mean_acr_x = (acr_gc[:, 1:9, 1:9, :-2] + acr_gc[:, 1:9, 1:9, 2:]) / 2.
    sel = np.abs(acr_gc[:, 1:9, 1:9, 1:9] / mean_acr_x - 1.0) > del_acr
    print(sel.shape, acr_gc[:, 1:9, 1:9, 1:9].shape, mean_acr_x.shape )
    acr_gc[:, 1:9, 1:9, 1:9][sel] = mean_acr_x[sel]
    
    # Smooth in y
    mean_acr_y = (acr_gc[:, 1:9, :-2, 1:9] + acr_gc[:, 1:9, 2:, 1:9]) / 2.
    sel = acr_gc[:, 1:9, 1:9, 1:9] / mean_acr_y - 1.0 > del_acr
    acr_gc[:, 1:9, 1:9, 1:9][sel] = mean_acr_y[sel]

    return acr_gc[:, 1:9, 1:9, 1:9]


def determine_grad(pres_gc, gid, pos, axis, use_gc_for_no_neigh=False, debug=False):
    # Default slice for dataset with GC
    # Selects actual data
    sl_gc = [
        slice(1, 9, None),
        slice(1, 9, None),
        slice(1, 9, None)
    ]
    
    # Default slice for datasets without GC
    sl = [
        slice(None, None, None),
        slice(None, None, None),
        slice(None, None, None)
    ]
    
    # Initialize field to 0
    pgrad = np.zeros_like(pres_gc[:, *sl_gc])
    
    # Get gradient using central difference using cell + gc of block
    sl_prev = sl_gc.copy()
    sl_curr = sl_gc.copy()
    sl_next = sl_gc.copy()
    sl_prev[2 - axis] = slice(None, -2, None)
    sl_curr[2 - axis] = slice(1, -1, None)
    sl_next[2 - axis] = slice(2, None, None)

    # Determine gradient for all blocks using GCs.
    # Assume unequal spacing and thus take average between forward and backward difference.
    # Automatically adjusts for blocks which do not have a neighbour and also reduces
    # to default central difference on uniform spacing.
    # pgrad = 0.5 * ((p_next - p_curr) / (x_next - x_curr) + (p_curr - p_prev) / (x_curr - x_prev))
    #    for uniform grid x_next - x_curr = x_curr - x_prev = dx
    # pgrad = 0.5 * (p_next - p_prev) / dx <-- typical central difference
    pgrad[...] = (pres_gc[:, *sl_next] - pres_gc[:, *sl_curr]) / (pos[:, *sl_next] - pos[:, *sl_curr])
    pgrad[...] += (pres_gc[:, *sl_curr] - pres_gc[:, *sl_prev]) / (pos[:, *sl_curr] - pos[:, *sl_prev])
    pgrad[...] /= 2.
    
    # pgrad[...] = (pres_gc[:, *sl_next] - pres_gc[:, *sl_prev]) / (pos[:, *sl_next] - pos[:, *sl_prev])
    # pgrad[...] = (pres_gc[:, *sl_next] - pres_gc[:, *sl_prev]) / (2 * dx[:, axis][:, None, None, None])
    
    # Treat special cases
    sl_prev_gc_m = sl_gc.copy()
    sl_next_gc_m = sl_gc.copy()
    sl_cur_m = sl.copy()
    # Forward difference
    sl_prev_gc_m[2 - axis] = 1 # slice(1, 2, None)
    sl_next_gc_m[2 - axis] = 2 # slice(2, 3, None)
    sl_cur_m[2 - axis] = 0 #  slice(0, 1, None)

    sl_prev_gc_p = sl_gc.copy()
    sl_next_gc_p = sl_gc.copy()
    sl_cur_p = sl.copy()
    # Backward difference
    sl_prev_gc_p[2 - axis] = -3 # slice(-3, -2, None)
    sl_next_gc_p[2 - axis] = -2 # slice(-2, -1, None)
    sl_cur_p[2 - axis] = -1 # slice(-1, None, None)

    # Update faces which are at the domain boundary
    # using forward / backward difference scheme
    at_boundary_m = gid[:, axis * 2] == -39
    at_boundary_p = gid[:, axis * 2 + 1] == -39
    
    if debug:
        print('-39')
        # print(pgrad[at_boundary_m].shape, pres_gc[at_boundary_m].shape, dx[at_boundary_m, axis].shape)
        print(sl_cur_m, sl_prev_gc_m, sl_next_gc_m)
        print()
        # print(pgrad[at_boundary_p].shape, pres_gc[at_boundary_p].shape, dx[at_boundary_p, axis].shape)
        print(sl_cur_p, sl_prev_gc_p, sl_next_gc_p)

    pgrad[at_boundary_m, *sl_cur_m] = (pres_gc[at_boundary_m, *sl_next_gc_m] - pres_gc[at_boundary_m, *sl_prev_gc_m])
    pgrad[at_boundary_m, *sl_cur_m] /= pos[at_boundary_m, *sl_next_gc_m] - pos[at_boundary_m, *sl_prev_gc_m]
    # pgrad[at_boundary_m, *sl_cur_m] /= dx[at_boundary_m, axis][:, None, None]

    pgrad[at_boundary_p, *sl_cur_p] = (pres_gc[at_boundary_p, *sl_next_gc_p] - pres_gc[at_boundary_p, *sl_prev_gc_p])
    pgrad[at_boundary_p, *sl_cur_p] /= pos[at_boundary_p, *sl_next_gc_p] - pos[at_boundary_p, *sl_prev_gc_p]
    # pgrad[at_boundary_p, *sl_cur_p] /= dx[at_boundary_p, axis][:, None, None]
    
    # return pgrad
    
    # Update faces which have no neighbour and data is from parent block.
    # When using central difference we need to account for
    # the extra cell spacing from the parent block of the neighbour.
    at_boundary_m = gid[:, axis * 2] == -1
    at_boundary_p = gid[:, axis * 2 + 1] == -1
    
    if use_gc_for_no_neigh:
        # Use central difference with unequal spacing
        sl_cur_gc_m = sl_gc.copy()
        sl_cur_gc_m[2 - axis] = 1 # slice(1, 2, None)
        
        sl_next_gc_m[2 - axis] = 2 # slice(2, 3, None)
        sl_prev_gc_m[2 - axis] = 0 # slice(0, 1, None)
        
        # Distance to prev is 2 times the local cell spacing
        # dx23 = dx, dx12 = 3 * dx, a = dx12 / dx23 = 3
        # (a * f_next + (1 - a) * f_cur - f_prev) / (2 * a * dx23)
        # = (a * f_next + (1 - a) * f_cur - f_prev) / (2 * a * dx)
        if debug:
            print('-1')
            print('Current target', sl_cur_m, '\tValue (axis)', sl_cur_m[2 - axis])
            print('Prev val', sl_prev_gc_m, '\tValue (axis)', sl_prev_gc_m[2 - axis])
            print('Curr val', sl_cur_gc_m, '\tValue (axis)', sl_cur_gc_m[2 - axis])
            print('Next val', sl_next_gc_m, '\tValue (axis)', sl_next_gc_m[2 - axis])
            print()
        
        a = 1.5
        #pgrad[at_boundary_m, *sl_cur_m] = pres_gc[at_boundary_m, *sl_next_gc_m] * a
        #pgrad[at_boundary_m, *sl_cur_m] += pres_gc[at_boundary_m, *sl_cur_gc_m] * (1 - a)
        #pgrad[at_boundary_m, *sl_cur_m] -= pres_gc[at_boundary_m, *sl_prev_gc_m]
        #pgrad[at_boundary_m, *sl_cur_m] /= 2 * dx[at_boundary_m, axis][:, None, None] * a
        
        if debug:
            print(
                pgrad[at_boundary_m, *sl_cur_m].shape,
                pres_gc[at_boundary_m, *sl_next_gc_m].shape,
                pres_gc[at_boundary_m, *sl_prev_gc_m].shape,
                # dx[at_boundary_m, axis][:, None, None].shape
            )
            
        pgrad[at_boundary_m, *sl_cur_m] = pres_gc[at_boundary_m, *sl_next_gc_m]
        pgrad[at_boundary_m, *sl_cur_m] -= pres_gc[at_boundary_m, *sl_prev_gc_m]
        # pgrad[at_boundary_m, *sl_cur_m] /= dx[at_boundary_m, axis][:, None, None] * (a + 1)
        pgrad[at_boundary_m, *sl_cur_m] /= pos[at_boundary_m, *sl_next_gc_m] - pos[at_boundary_m, *sl_prev_gc_m]

        sl_cur_gc_p = sl_gc.copy()
        sl_cur_gc_p[2 - axis] = -2 # slice(-2, -1, None)
    
        sl_prev_gc_p[2 - axis] = -3 # slice(-3, -2, None)
        sl_next_gc_p[2 - axis] = -1 # slice(-1, None, None)
        
        if debug:
            print('Current target', sl_cur_p, '\tValue (axis)', sl_cur_p[2 - axis])
            print('Prev val', sl_prev_gc_p, '\tValue (axis)', sl_prev_gc_p[2 - axis])
            print('Curr val', sl_cur_gc_p, '\tValue (axis)', sl_cur_gc_p[2 - axis])
            print('Next val', sl_next_gc_p, '\tValue (axis)', sl_next_gc_p[2 - axis])

        # Distance to next is 2 times the local cell spacing
        # dx23 = 1.5 * dx, dx12 = dx, a = dx12 / dx23 = 2. / 3. --> dx23 = 1.5 * dx = dx / a
        # (a * f_next + (1 - a) * f_cur - f_prev) / (2 * a * dx23)
        # = (a * f_next + (1 - a) * f_cur - f_prev) / (2 * dx)
        #a = 1. / a
        #pgrad[at_boundary_p, *sl_cur_p] = pres_gc[at_boundary_p, *sl_next_gc_p] * a
        #pgrad[at_boundary_p, *sl_cur_p] += pres_gc[at_boundary_p, *sl_cur_gc_p] * (1 - a)
        #pgrad[at_boundary_p, *sl_cur_p] -= pres_gc[at_boundary_p, *sl_prev_gc_p]
        #pgrad[at_boundary_p, *sl_cur_p] /= 2 * dx[at_boundary_p, axis][:, None, None]
        
        pgrad[at_boundary_p, *sl_cur_p] = pres_gc[at_boundary_p, *sl_next_gc_p]
        pgrad[at_boundary_p, *sl_cur_p] -= pres_gc[at_boundary_p, *sl_prev_gc_p]
        # pgrad[at_boundary_p, *sl_cur_p] /= dx[at_boundary_p, axis][:, None, None] * (a + 1)
        pgrad[at_boundary_p, *sl_cur_p] /= pos[at_boundary_p, *sl_next_gc_p] - pos[at_boundary_p, *sl_prev_gc_p]

    else:
        if debug:
            print('-1')
            print('Current target', sl_cur_m, '\tValue (axis)', sl_cur_m[2 - axis])
            print('Prev val', sl_prev_gc_m, '\tValue (axis)', sl_prev_gc_m[2 - axis])
            print('Next val', sl_next_gc_m, '\tValue (axis)', sl_next_gc_m[2 - axis])
            print()
            print('Current target', sl_cur_p, '\tValue (axis)', sl_cur_p[2 - axis])
            print('Prev val', sl_prev_gc_p, '\tValue (axis)', sl_prev_gc_p[2 - axis])
            print('Next val', sl_next_gc_p, '\tValue (axis)', sl_next_gc_p[2 - axis])
            
        # Use backward/forward difference without guard cells
        pgrad[at_boundary_m, *sl_cur_m] = pres_gc[at_boundary_m, *sl_next_gc_m] - pres_gc[at_boundary_m, *sl_prev_gc_m]
        # pgrad[at_boundary_m, *sl_cur_m] /= dx[at_boundary_m, axis][:, None, None]
        pgrad[at_boundary_m, *sl_cur_m] /= pos[at_boundary_m, *sl_next_gc_m] - pos[at_boundary_m, *sl_prev_gc_m]
        
        pgrad[at_boundary_p, *sl_cur_p] = pres_gc[at_boundary_p, *sl_next_gc_p] - pres_gc[at_boundary_p, *sl_prev_gc_p]
        # pgrad[at_boundary_p, *sl_cur_p] /= dx[at_boundary_p, axis][:, None, None]
        pgrad[at_boundary_p, *sl_cur_p] /= pos[at_boundary_p, *sl_next_gc_p] - pos[at_boundary_p, *sl_prev_gc_p]

    return pgrad

# Determine acceleration from magnetic pressure acting on the gas
def acc_mag_pres(pres_gc, dens, bbox, gid):
    # Get cell size in each block
    dx = (bbox[:, :, 1] - bbox[:, :, 0]) / dens.shape[1:]
    
    pgradx = determine_grad(
        pres_gc=pres_gc, gid=gid, # dx=dx,
        axis=0, use_gc_for_no_neigh=True
    )
    pgrady = determine_grad(
        pres_gc=pres_gc, gid=gid, # dx=dx,
        axis=1, use_gc_for_no_neigh=True
    )
    pgradz = determine_grad(
        pres_gc=pres_gc, gid=gid, # dx=dx,
        axis=2, use_gc_for_no_neigh=True
    )
    
    # Determine acting acceleration from pressure gradient
    acx_p = -pgradx / dens
    acy_p = -pgrady / dens
    acz_p = -pgradz / dens

    return acx_p, acy_p, acz_p
