import healpy
import scipy.constants as const
import numpy as np
import matplotlib.pyplot as plt
import h5py
import flash_amr_tools as amr_tools


# Determine corner coordinate from center coordinate and index
# Arguments:
#   cx, cy, cz  - cell center positions
#   dx, dy, dz  - cell sizes
#   nr          - id of corner
def corner(cx, cy, cz, dx, dy, dz, nr):

    cx += dx * 0.5 * (2 * (nr%2) - 1)
    cy += dy * 0.5 * (2 * ((nr//2)%2) - 1)
    cz += dz * 0.5 * (2 * ((nr//4)%2) - 1)

    return cx, cy, cz

# Function to create a healpix map
# Arguments:
#   tr filter   - Cell filter used for density
#   data        - Density for now
#   bs          - Block size of each block
#   rlvl        - Refinement level of each block
#   cpos        - Cell positions (in relative coordinates)
#   corners     - Flag to use corner positions instead of cell centers
def create_hpmap(tr_filter, data, bs, rlvl, cpos, corners=False, nest=True):
    # Number of iterations to do
    # 1 for cell centers
    # 6 for all corners of the cell
    n_iter = 1
    if corners:
        n_iter = 8
    
    # Determine cell size of all blocks
    dx = bs / 8.
    
    # Determine relative distances of all cells     
    r = (cell_pos**2).sum(axis=4)**0.5
    # Get max distance to determine maximum resolution
    r_max = r.max()
    
    # Determine unique refinement levels
    n_rlvl = np.unique(rlvl)
    
    # Maximum resolution is calcuclated by calculating high we need to resolve
    # to represent on cell at maximum resolution at the furthes distance.
    # max_res = A_sphere / smallest cell surface / 12 (lowest number of healpixel)
    max_res = np.sqrt((4 * np.pi * r_max**2) / dx.min()**2 / 12)
    # To allow regriding the the lower resolution images we find 
    # the next largest power of 2 as 1 healpixel is split up to 4 from one
    # refinement level to the next.
    max_res = int(2**np.ceil(np.log2(max_res)))
    
    # Create 1D array for all the pixels of the map at max resolution 
    hpmap = np.zeros(healpy.nside2npix(max_res))

    # Loop over all cell resolution level
    for lvl in n_rlvl:
        # Loop over all healpix resolution levels
        for i in range(int(np.log2(max_res))+1):
            # Create copy of filter to add distance filter
            c_true = tr_filter.copy()
            
            # Filter for cells at right cell resolution level
            c_true *= (rlvl == lvl)[:, None, None, None]

            # Skip loop if no cell fulfills both conditions
            if np.sum(c_true) == 0:
                continue

            # Determine current resolution (npixel)
            # Does not include the factor 12
            t_res = 2 ** (i)

            # Determine the number of healpixel of the current and next
            # refinement level
            n_min = 12 * (2**i)**2
            n_max = 12 * (2**(i+1))**2

            # From these determine the minimum and maximum distance of cells
            # we need to consider.
            # So to say which cells are smaller than the current angular resolution
            # but greater than the higher refinement level.
            # d = (n_min * A_cell / 4 / pi)**0.5
            
            x_min = np.sqrt(n_min * dx[rlvl == lvl].min()**2 / 4 / np.pi)
            x_max = np.sqrt(n_max * dx[rlvl == lvl].min()**2 / 4 / np.pi)

            print(t_res, x_min, x_max)
            # Update filter to select cells which are in the shell.
            c_true *= r > x_min
            c_true *= r <= x_max
            
            # Only if cell are in the shell determine their healpixel positions
            if np.any(c_true):
                # Loop over either cell centers or corners
                for j in range(n_iter):
                    tdx = dx[rlvl == lvl].min(axis=0)
                    # Get corner position if corners are selected
                    if corners:
                        tx, ty, tz = corner(
                            cx=cpos[..., 0][c_true].copy(),
                            cy=cpos[..., 1][c_true].copy(),
                            cz=cpos[..., 2][c_true].copy(),
                            dx=tdx[0], dy=tdx[1], dz=tdx[2],
                            nr=j
                        )

                    else:
                        tx = cpos[..., 0][c_true]
                        ty = cpos[..., 1][c_true]
                        tz = cpos[..., 2][c_true]

                    # Determine pixel index of each cell at current resolution 
                    hpix = healpy.vec2pix(t_res, tx, ty, tz, nest=nest)

                    # Area of a healpixel
                    ar = 4 * np.pi * x_min.astype(np.float64)**2 / n_min
                    # Volume of one cell at current cell size
                    vol = np.prod(dx[rlvl == lvl].astype(np.float64).min(axis=0))
                    # Effective depth to get mass in cell
                    # And thus contribution to column density
                    d_eff = vol / ar
                    
                    # Adapt d_eff to include factor 1/8 to counter
                    # overcounting due to usage of corners
                    if corners:
                        d_eff /= 8.

                    # Create map of current healpix resolution
                    tmap = np.zeros(healpy.nside2npix(t_res))
                    # Add effective column density onto the grid
                    np.add.at(tmap, hpix, dens[c_true] * d_eff)
                    
                    # Add contrbution to max healpix resolution map.
                    if nest:
                      hpmap += healpy.ud_grade(map_in=tmap, nside_out=max_res, order_in='NESTED', order_out='NESTED')
                    else:
                      hpmap += healpy.ud_grade(map_in=tmap, nside_out=max_res)
    return hpmap


# Set file name
fname = ''

# Set corner coordinates of region to consider
# Either use full domain
xmin = []
xmax = []
# Or select subregion
# xmin = np.array([ 2.8931249e+20, -5.7862501e+20, -1.9287499e+20])
# xmax = np.array([ 6.7506249e+20, -1.9287501e+20,  1.9287499e+20])

# Define position from which to look
view_pos = np.asarray([0., 0., 0.])
view_pos[2] += 1. * const.kilo * const.parsec / const.centi
# view_pos = np.asarray([4.72390664e20, -3.503393555e20, 7.534179687e17])



# Get all block which are part of the above defined region
blist, brefs, bns = amr_tools.get_true_blocks(
    fname, xmin, xmax
)

# Open file
pf = h5py.File(fname)

# Select dataset and filter for block which are in above defined region
dens = pf['dens'][()][blist]
vel_vec = [pf['velx'][()][blist], pf['vely'][()][blist], pf['velz'][()][blist]]

# Helpful other data sets
bbox = pf['bounding box'][()][blist]
bs = pf['block size'][()][blist]
ref_lvl = pf['refine level'][()][blist]

# Get corner coordinates of blocks
# These can slightly deviate from xmin and xmax which is set above
low_cor, up_cor = (bbox[0, :, 0], bbox[-1, :, 1])

# Determine cell size
# Total length / number of blocks in each direction
#              / number of blocks for each refinement over base
#              / number of cells in each block (per direction)
dx = (up_cor - low_cor) / bns / 2**(brefs[0] - brefs[1] + 3)


# Determine all cell positions
# Create emtpy array for all cells
cell_pos = np.zeros(tuple(list(dens.shape) + [3]))

# Loop over all blocks
for i in range(blist.size):
    # Get corner positions of current block
    temp_low_cor, temp_up_cor = bbox[i, :, 0], bbox[i, :, 1]
    # Get cell size of current block
    tdx = (temp_up_cor - temp_low_cor) / 8.
    
    # Determine cell center positions
    tx, ty, tz = np.meshgrid(
        np.linspace(temp_low_cor[0], temp_up_cor[0], 8, endpoint=False) + tdx[0] / 2.,
        np.linspace(temp_low_cor[1], temp_up_cor[1], 8, endpoint=False) + tdx[1] / 2.,
        np.linspace(temp_low_cor[2], temp_up_cor[2], 8, endpoint=False) + tdx[2] / 2.,
        indexing='ij'
    )
    
    # Save values in array
    cell_pos[i, ..., 0] = tx.T
    cell_pos[i, ..., 1] = ty.T
    cell_pos[i, ..., 2] = tz.T
    
    # Clear temporary arrays
    del tx, ty, tz

# To get adopt a shift to a certain object one only needs to subtract the position
# from all coordinates
cell_pos -= view_pos[None, None, None, None, :]


# Plotting
# This could be used for pre selecting cells on certain criteria
tr = np.ones_like(dens, dtype=bool)

# Get healpy map using cell centers
mmap = create_hpmap(tr_filter=tr, data=dens, bs=bs, rlvl=ref_lvl, cpos=cell_pos, corners=False)
# Get healpy map using cell corners (smoother images)
mmap_cor = create_hpmap(tr_filter=tr, data=dens, bs=bs, rlvl=ref_lvl, cpos=cell_pos, corners=True)


fig = plt.figure()
healpy.mollview(
    np.log10(mmap / const.kilo / const.m_p),
    rot=0, title='Column density map', unit=r'N [cm$^{-2}$]',
    fig=fig.number, nest=True
)
#healpy.graticule()
fig.show()

fig = plt.figure()
healpy.mollview(
    np.log10(mmap_cor / const.kilo / const.m_p),
    rot=180, title='Column density map using corners', unit=r'N [cm$^{-2}$]',
    fig=fig.number
)
healpy.graticule()
fig.show()


