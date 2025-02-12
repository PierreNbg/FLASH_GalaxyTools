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
    zz = ccoords[..., 2]
    
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
    # Update pressure to be uniform in mid-plane
    # p_cor = pres.max() - pres[:,:,pres.shape[2]//2]
    
    # pnew = pres + p_cor[:,:,None]
    
    ccoords = get_cell_centers(bbox)
    
    # Separate components
    xx = ccoords[..., 0]
    yy = ccoords[..., 1]
    zz = ccoords[..., 2]
    
    # Get cylindrical radius
    rr = (xx ** 2 + yy ** 2) ** 0.5
    
    # Get cell size in each block
    dx = (bbox[:, :, 1] - bbox[:, :, 0]) / dens.shape[1:]
    
    # Initialize fields to 0
    # of grad P (Hydrostatic equilibrium)
    pgradx = np.zeros_like(pres_gc[:, 1:9, 1:9, 1:9])
    pgrady = np.zeros_like(pgradx)
    
    # Determine central difference for x and y component
    pgradx[...] = (pres_gc[:, 1:9, 1:9, 2:] - pres_gc[:, 1:9, 1:9, :-2]) / (2 * dx[:, 0][:, None, None, None])
    pgrady[...] = (pres_gc[:, 1:9, 2:, 1:9] - pres_gc[:, 1:9, :-2, 1:9]) / (2 * dx[:, 1][:, None, None, None])
    
    # Only use forward / backward difference for cells at domain boundary
    # in x-direction
    at_xboundary_m = gid[:, 0] == -39
    at_xboundary_p = gid[:, 1] == -39
    
    pgradx[at_xboundary_m, :, :, 0] = (pres_gc[at_xboundary_m, 1:9, 1:9, 2] - pres_gc[at_xboundary_m, 1:9, 1:9, 1])
    pgradx[at_xboundary_m, :, :, 0] /= dx[at_xboundary_m, 0][:, None, None]
    
    pgradx[at_xboundary_p, :, :, -1] = (pres_gc[at_xboundary_p, 1:9, 1:9, -2] - pres_gc[at_xboundary_p, 1:9, 1:9, -3])
    pgradx[at_xboundary_p, :, :, -1] /= dx[at_xboundary_p, 0][:, None, None]
    
    # in y-direction
    at_yboundary_m = gid[:, 2] == -39
    at_yboundary_p = gid[:, 3] == -39
    
    pgrady[at_yboundary_m, :, 0, :] = (pres_gc[at_yboundary_m, 1:9, 2, 1:9] - pres_gc[at_yboundary_m, 1:9, 1, 1:9])
    pgrady[at_yboundary_m, :, 0, :] /= dx[at_yboundary_m, 1][:, None, None]
    
    pgrady[at_yboundary_p, :, -1, :] = (pres_gc[at_yboundary_p, 1:9, -2, 1:9] - pres_gc[at_yboundary_p, 1:9, -3, 1:9])
    pgrady[at_yboundary_p, :, -1, :] /= dx[at_yboundary_p, 1][:, None, None]

    # Still some weird bug at boundaries of refinement levels
    # Recalculate at boundaries if neighbour does not exist.
    # Weigh guard cell from parent with 0.5 as their real distance is 2 times the
    # local cell spacing
    at_xboundary_m = gid[:, 0] == -1
    at_xboundary_p = gid[:, 1] == -1
    
    pgradx[at_xboundary_m, :, :, 0] = (pres_gc[at_xboundary_m, 1:9, 1:9, 2] - pres_gc[at_xboundary_m, 1:9, 1:9, 1]) * 1.0
    #pgradx[at_xboundary_m, :, :, 0] += (pres_gc[at_xboundary_m, 1:9, 1:9, 1] - pres_gc[at_xboundary_m, 1:9, 1:9, 0]) / 1.5
    #pgradx[at_xboundary_m, :, :, 0] /= 2 * dx[at_xboundary_m, 0][:, None, None]
    pgradx[at_xboundary_m, :, :, 0] /= dx[at_xboundary_m, 0][:, None, None]
    
    pgradx[at_xboundary_p, :, :, -1] = (pres_gc[at_xboundary_p, 1:9, 1:9, -2] - pres_gc[at_xboundary_p, 1:9, 1:9, -3]) * 1.0
    #pgradx[at_xboundary_p, :, :, -1] += (pres_gc[at_xboundary_p, 1:9, 1:9, -1] - pres_gc[at_xboundary_p, 1:9, 1:9, -2]) / 1.5
    #pgradx[at_xboundary_p, :, :, -1] /= 2 * dx[at_xboundary_p, 0][:, None, None]
    pgradx[at_xboundary_p, :, :, -1] /= dx[at_xboundary_p, 0][:, None, None]
    
    # in y-direction
    at_yboundary_m = gid[:, 2] == -1
    at_yboundary_p = gid[:, 3] == -1
    
    pgrady[at_yboundary_m, :, 0, :] = (pres_gc[at_yboundary_m, 1:9, 2, 1:9] - pres_gc[at_yboundary_m, 1:9, 1, 1:9]) * 1.0
    #pgrady[at_yboundary_m, :, 0, :] += 1.0 * (pres_gc[at_yboundary_m, 1:9, 1, 1:9] - pres_gc[at_yboundary_m, 1:9, 0, 1:9])
    #pgrady[at_yboundary_m, :, 0, :] /= 2 * dx[at_yboundary_m, 1][:, None, None] / 2
    pgrady[at_yboundary_m, :, 0, :] /= dx[at_yboundary_m, 1][:, None, None]
    
    pgrady[at_yboundary_p, :, -1, :] = (pres_gc[at_yboundary_p, 1:9, -2, 1:9] - pres_gc[at_yboundary_p, 1:9, -3, 1:9]) * 1.0
    #pgrady[at_yboundary_p, :, -1, :] += 1.0 * (pres_gc[at_yboundary_p, 1:9, -1, 1:9] - pres_gc[at_yboundary_p, 1:9, -2, 1:9])
    #pgrady[at_yboundary_p, :, -1, :] /= 2 * dx[at_yboundary_p, 1][:, None, None] / 2
    pgrady[at_yboundary_p, :, -1, :] /= dx[at_yboundary_p, 1][:, None, None]
    
    
    # Use hydrostatic equilibrium to get acceleration due to pressure profile
    acx_p = -pgradx / dens
    acx_gc = create_ghostcells(acx_p)
    acx_gc = fill_ghostcells(data_pad=acx_gc, gid=gid, bbox=bbox)

    at_xboundary_m = gid[:, 0] == -1
    at_xboundary_p = gid[:, 1] == -1

    acx_gc[at_xboundary_m, 1:9, 1:9, 1] = 1.5 * dx[at_xboundary_m, 0][:, None, None] * acx_gc[at_xboundary_m, 1:9, 1:9, 0]
    acx_gc[at_xboundary_m, 1:9, 1:9, 1] += dx[at_xboundary_m, 0][:, None, None] * acx_gc[at_xboundary_m, 1:9, 1:9, 2]
    acx_gc[at_xboundary_m, 1:9, 1:9, 1] /= 2.5 * dx[at_xboundary_m, 0][:, None, None]
    
    acx_gc[at_xboundary_p, 1:9, 1:9, -2] = 1.5 * dx[at_xboundary_p, 0][:, None, None] * acx_gc[at_xboundary_p, 1:9, 1:9, -1]
    acx_gc[at_xboundary_p, 1:9, 1:9, -2] += dx[at_xboundary_p, 0][:, None, None] * acx_gc[at_xboundary_p, 1:9, 1:9, -3]
    acx_gc[at_xboundary_p, 1:9, 1:9, -2] /= 2.5 * dx[at_xboundary_p, 0][:, None, None]
    
    acx_p = acx_gc[:, 1:9, 1:9, 1:9]
    del acx_gc

    acy_p = -pgrady / dens
    acy_gc = create_ghostcells(acy_p)
    acy_gc = fill_ghostcells(data_pad=acy_gc, gid=gid, bbox=bbox)

    at_yboundary_m = gid[:, 2] == -1
    at_yboundary_p = gid[:, 3] == -1
    
    acy_gc[at_yboundary_m, 1:9, 1, 1:9] = 1.5 * dx[at_yboundary_m, 1][:, None, None] * acy_gc[at_yboundary_m, 1:9, 0, 1:9]
    acy_gc[at_yboundary_m, 1:9, 1, 1:9] += dx[at_yboundary_m, 1][:, None, None] * acy_gc[at_yboundary_m, 1:9, 2, 1:9]
    acy_gc[at_yboundary_m, 1:9, 1, 1:9] /= 2.5 * dx[at_yboundary_m, 1][:, None, None]
    
    acy_gc[at_yboundary_p, 1:9, -2, 1:9] = 1.5 * dx[at_yboundary_p, 1][:, None, None] * acy_gc[at_yboundary_p, 1:9, -1, 1:9]
    acy_gc[at_yboundary_p, 1:9, -2, 1:9] += dx[at_yboundary_p, 1][:, None, None] * acy_gc[at_yboundary_p, 1:9, -3, 1:9]
    acy_gc[at_yboundary_p, 1:9, -2, 1:9] /= 2.5 * dx[at_yboundary_p, 1][:, None, None]

    acy_p = acy_gc[:, 1:9, 1:9, 1:9]
    del acy_gc

    # Get angle of radial vector
    phi = np.arctan2(yy, xx)
    
    # Get radial component of total acceleration (gas+ext+pres)
    # ac* = (gas + ext), ac*_p = (pressure)
    # acr = (acx_p + acx) * np.cos(phi) + (acy_p + acy) * np.sin(phi)
    acr = (acx + acx_p) * np.cos(phi) + (acy + acy_p) * np.sin(phi)
    
    acr = -acr
    # Get absolute rotation velocity
    # from dark matter acceleration
    # v = (r * d/dr phi)**0.5 = (-r * acc)**0.5
    v_abs = (acr * rr) ** 0.5
    v_abs = np.nan_to_num(v_abs)
    
    print('Rotation after correction for radial pressure gradient')
    print('\tVelocity range (km/s) (min/max):\t', v_abs.min() / 1e5, v_abs.max() / 1e5)
    
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
