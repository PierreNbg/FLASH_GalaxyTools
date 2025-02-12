import scipy.constants as const

# Solar mass in g (same as constants.f90)
m_solar = 1.98847e33

# Distance conversion
pc2cm = const.parsec / const.centi
kpc2cm = const.kilo * const.parsec / const.centi

# Time conversion
yr2sec = const.year
kyr2sec = const.kilo * const.year
myr2sec = const.mega * const.year

# Gravitational constant in cgs
g_newt = const.kilo * const.gravitational_constant