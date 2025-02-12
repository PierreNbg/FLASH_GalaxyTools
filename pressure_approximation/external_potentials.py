import numpy as np
from .constants import *
from .io_amr import read_settings

# Dark matter profile Burkert et al
def get_g_dm(x, y, z):
    # Get cell coordinates
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    # Determine distance to center of galaxy
    rr = (xx ** 2 + yy ** 2 + zz ** 2) ** 0.5
    
    # Get settings defined in settings.txt
    sim_set = read_settings('settings.txt')
    # Dark matter mass [g]
    m0 = sim_set['dm_mass'] * m_solar
    print('DM mass: ', m0)
    # kpc to cm conversion factor
    kpc2cm = const.kilo * const.parsec / const.centi
    # Gravitational constant in cgs
    g_newt = const.kilo * const.gravitational_constant
    
    # r0 in cm
    r0 = 3.07 * kpc2cm * (m0 / (m_solar * 1e9)) ** (3. / 7.)
    # rhod0 in g cm^-3
    rhod0 = 1.46e-24 * (m0 / (m_solar * 1e9)) ** (-2 / 7.)
    konst = np.pi * g_newt * rhod0 * r0 ** 2
    
    # Calculate the absolute acceleration
    g = -(rr ** 2 + r0 * rr) / (r0 ** 2 + rr ** 2)
    g += np.arctan(rr / r0)
    g += (rr ** 2 + r0 * rr) / (r0 ** 2 + rr * r0)
    g -= np.log(1. + rr / r0)
    g -= (rr ** 3 - r0 * rr ** 2) / (r0 ** 3 + r0 * rr ** 2)
    g -= 0.5 * np.log(1. + (rr / r0) ** 2)
    g *= -2 * konst * r0 / rr ** 3
    
    # Remove nan that can occur for rr = 0
    g = np.nan_to_num(g)
    
    # Get acceleration for each component
    accx_dm = -g * xx
    accy_dm = -g * yy
    accz_dm = -g * zz
    
    return accx_dm, accy_dm, accz_dm


# Old stellar population Miyamoto & Nagai
def get_g_stellar(x, y, z):
    # Get cell coordinates
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    # Determine distance to center of galaxy
    rr = (xx ** 2 + yy ** 2) ** 0.5
    
    # Read settings for mass and scaling factors
    sim_set = read_settings('settings.txt')
    
    # Mass of stellar population
    mass = sim_set['stellar_mass'] * m_solar
    # Scaling in radius
    a = sim_set['a_star'] * const.parsec / const.centi
    # Scaling in height
    b = sim_set['b_star'] * const.parsec / const.centi
    
    # Gravitational constant in cgs
    g_newt = const.kilo * const.gravitational_constant
    
    # Pre-factor
    konst = g_newt * mass
    
    # Common component of acceleration
    g = konst * (rr ** 2 + (a + (zz ** 2 + b ** 2) ** 0.5) ** 2) ** (-1.5)
    
    # Get acceleration for each component
    accx_star = -g * xx
    accy_star = -g * yy
    accz_star = -g * zz * (1. + a / (zz ** 2 + b ** 2) ** 0.5)
    
    return accx_star, accy_star, accz_star


# Get rotation from given acceleration
# Assumes ration around z-axis
# Only radial (cylindrical) acceleration of x- and y-component are considered
def get_dm_rot(x, y, z, acx_dm, acy_dm, acz_dm, return_vel=False, cut_reg=None):
    # Get cell centers
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    # Determine spherical distance to center
    # rr = (xx**2 + yy**2 + zz**2)**0.5
    rr = (xx ** 2 + yy ** 2) ** 0.5
    
    phi = np.arctan2(yy, xx)
    acr = acx_dm * np.cos(phi) + acy_dm * np.sin(phi)
    a_abs = -acr
    # Get absolute rotation velocity
    # from dark matter acceleration
    # v = (r * d/dr phi)**0.5 = (-r * acc)**0.5
    # v_abs = (-acx_dm * xx - acy_dm * yy - acz_dm * zz)**0.5
    # a_abs = (acx_dm**2 + acy_dm**2)**0.5
    
    # v_abs = (-acx_dm * xx - acy_dm * yy)**0.5
    v_abs = (a_abs * rr) ** 0.5
    
    # alpha = np.arctan2(yy, xx)
    
    # Determine x and y components of rotation
    # vel_x = -v_abs * yy / rr
    # vel_y = v_abs * xx / rr
    
    vel_x = -v_abs * np.sin(phi)
    vel_y = v_abs * np.cos(phi)
    
    # Remove nan from rr = 0
    vel_x = np.nan_to_num(vel_x)
    vel_y = np.nan_to_num(vel_y)
    
    # Cylindrical radius
    # needed for centripetal force calculation
    r = (xx ** 2 + yy ** 2) ** 0.5
    
    # Absolute acceleration from centripetal force
    acc_rot = (vel_x ** 2 + vel_y ** 2) / r
    # Remove nan from r = 0
    acc_rot = np.nan_to_num(acc_rot)
    
    # Determine acceleration for x and y component
    # acx_rot = acc_rot * xx / r
    # acy_rot = acc_rot * yy / r
    
    acx_rot = acc_rot * np.cos(phi)
    acy_rot = acc_rot * np.sin(phi)
    
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
    
    if return_vel:
        return r, v_abs
    else:
        return (acx_rot, acy_rot, acz_rot), (vel_x, vel_y)

