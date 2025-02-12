import os
import numpy as np
import yt
from astropy.stats import sigma_clip
from yt.visualization.volume_rendering.api import Scene # , VolumeSource, PointSource
from yt.visualization.volume_rendering.render_source import KDTreeVolumeSource as VolumeSource
import h5py
import flash_amr_tools as amr_tools
import scipy.constants as const
import matplotlib.pyplot as plt
import sys
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colormaps



cm = yt.units.cm

# create custom colormap
blue_cm = LinearSegmentedColormap.from_list("myBlue", [(0, 0, 1), (0, 0, 1)], N=2)
green_cm = LinearSegmentedColormap.from_list("myGreen", [(0, 1, 0), (0, 1, 0)], N=2)
yell_cm = LinearSegmentedColormap.from_list("myYell", [(1, 1, 0), (1, 1, 0)], N=2)
red_cm = LinearSegmentedColormap.from_list("myRed", [(1, 0, 0), (1, 0, 0)], N=2)
dens_cm = LinearSegmentedColormap.from_list("myDens", [(1, 1, 1), (1, 1, 1)], N=2)

colormaps.register(cmap=blue_cm)
colormaps.register(cmap=green_cm)
colormaps.register(cmap=yell_cm)
colormaps.register(cmap=red_cm)
colormaps.register(cmap=dens_cm)


# Linear interpolation
def linramp(vals, minval, maxval):
    return (vals - vals.min()) / (vals.max() - vals.min())

def sqtramp(vals, minval, maxval):
    return (vals - vals.min())**0.5 / (vals.max() - vals.min())**0.5

def quadramp(vals, minval, maxval):
    return (vals - vals.min())**2 / (vals.max() - vals.min())**2


# Update all parents with data from their children
def update_all_parent_blocks(data, gid):
    # Get list of block ids
    # Going backwards through the list
    # will automatically fill all parents
    blist = np.arange(gid.shape[0])[::-1]
    
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


# General function to create a volume source in YT
def make_volumesource(
    field_name, ds,
    lower, upper,
    cmap, scale,
    log=True, nlayers=8,
    scale_func=linramp,
    opacity=False
):
    """Function to create a volume source in YT
    :param field_name: field name of the quantity to render (needs to be in ds)
    :param ds: dataset which contains the quantities
    :param lower: lower limit value of the color transfer function
    :param upper: upper limit value of the color transfer function
    :param cmap: colormap of the color transfer function
    :param scale: rescaling factor (boosting factor)
    :param log: boolean flag to change between linear and log scale (Default: True)
    :param nlayers: number of layers (not sure if needed, Default: 8)
    :param scale_func: scaling function (Default: linramp)
    :param opacity: Enable gray opacity (Default: False)
    :return: YT VolumeSource with transfer function included
    """
    source = VolumeSource(ds, field_name)
    source.set_field(field_name)
    source.set_log(log)
    bounds = (lower, upper)
    if log:
        tf = yt.ColorTransferFunction(np.log10(bounds))
        tf.add_layers(nlayers, colormap=cmap)
        tf.map_to_colormap(np.log10(lower), np.log10(upper), colormap=cmap, scale_func=linramp, scale=scale)
    else:
        tf = yt.ColorTransferFunction(bounds)
        tf.add_layers(nlayers, colormap=cmap)
        tf.map_to_colormap(lower, upper, colormap=cmap, scale_func=scale_func, scale=scale)
    source.tfh.tf = tf
    source.tfh.bounds = bounds
    source.tfh.grey_opacity = opacity
    source.tfh.plot('transfer_map_%s.png' % field_name, profile_field=field_name)
    return source


# Conversion factor from parsec to centimeter
pc2cm = const.parsec / const.centi

# Region of interest to cut from simulation
#if len(sys.argv) > 2:
#    xmin = -np.asarray([float(sys.argv[2]), float(sys.argv[2]), float(sys.argv[2])])
#    add_name = '_' + str(sys.argv[2])
#else:
#    xmin = np.asarray([-600., -600., -600.])
#    add_name = ''

#xmin *= pc2cm
#xmax = -xmin

xmin = [-6.17200022062524e+18, -6.17200022062524e+18, -6.17200022062524e+18]
xmax = [6.17200022062524e+18, 6.17200022062524e+18, 6.17200022062524e+18]

xmin = np.asarray(xmin) / 4
xmax = np.asarray(xmax) / 4

# Folder which contains the file
folder = '/projects/pnuernb1/DwarfGal/run_7'

# File name
#fname = 'DwarfGal_hdf5_plt_cnt_%04d'
# Add file number
#fname = fname % int(sys.argv[1])
fname = 'SpitzerTest_hdf5_plt_cnt_0230'

# Create full filename + path
#fname = os.path.join(folder, fname)

prefac = False

# Determine block list within defined region
blist, brefs, bns = amr_tools.get_true_blocks(fname, xmin, xmax, max_ref_given=8)
# Read data sets
pf = h5py.File(fname, 'r')
gid = pf['gid'][()]

# Get simulation time
time = dict(pf['real scalars'][()])[('%-80s' % 'time').encode('utf-8')]
time /= const.year * const.mega

# Preselect data sets and reduce to only necessary blocks
dens = pf['dens'][()]
dens = update_all_parent_blocks(dens, gid)
dens = dens[blist]

iha = pf['iha '][()]
iha = update_all_parent_blocks(iha, gid)
iha = iha[blist]

ihp = pf['ihp '][()]
ihp = update_all_parent_blocks(ihp, gid)
ihp = ihp[blist]

ih2 = pf['ih2 '][()]
ih2 = update_all_parent_blocks(ih2, gid)
ih2 = ih2[blist]

#temp = pf['temp'][()]
#temp = update_all_parent_blocks(temp, gid)[blist]

# Helper datasets
bbox = pf['bounding box'][()][blist]
xmin = bbox[0, :, 0]
xmax = bbox[-1, :, 1]

bs = pf['block size'][()][blist]
ref_lvl = pf['refine level'][()][blist]

# Sink list dataset
# Might need to change number of particles: 100
# Or number of properties: 88
#sinks = pf['sinkList'][()].reshape((100, 88))
# Select only active sinks (inactive sinks have all 0)
#sinks = sinks[np.logical_not(np.all(sinks == 0., axis=1))]
# Get sink positions
#sink_pos = sinks[:, :3]

# Project quantities onto uniform grid
# Maybe find way for amr grid to reduce memory footprint
print('Creating uniform grid for iha')
iha_grid = amr_tools.get_cube(
    data=dens * iha,
    bbox=bbox, bsize=bs, ref_lvl=ref_lvl,
    brefs=brefs, bns=bns
)
print('Creating uniform grid for ihp')
ihp_grid = amr_tools.get_cube(
    data=dens * ihp,
    bbox=bbox, bsize=bs, ref_lvl=ref_lvl,
    brefs=brefs, bns=bns
)
print('Creating uniform grid for ih2')
ih2_grid = amr_tools.get_cube(
    data=dens * ih2,
    bbox=bbox, bsize=bs, ref_lvl=ref_lvl,
    brefs=brefs, bns=bns
)

#temp_grid = amr_tools.get_cube(
#    data=temp,
#    bbox=bbox, bsize=bs, ref_lvl=ref_lvl,
#    brefs=brefs, bns=bns
#)

#dens_grid = amr_tools.get_cube(
#    data=dens,
#    bbox=bbox, bsize=bs, ref_lvl=ref_lvl,
#    brefs=brefs, bns=bns
#)

# Create dataset with uniform grids
ds = yt.load_uniform_grid(
    dict(
        h2=(ih2_grid, "g/cm**3"),
        ha=(iha_grid, "g/cm**3"),
        hp=(ihp_grid, "g/cm**3"),
        #dens=(dens_grid, "g/cm**3"),
        #t=(temp_grid, "K")
    ),
    ih2_grid.shape,
    bbox=np.asarray([xmin, xmax]).T,
    nprocs=4,
    length_unit='cm'
)

del iha_grid, ih2_grid, ihp_grid #, temp_grid, dens_grid

# Create new scene for render
sc = Scene()
sc.background_color = np.array([1., 1., 1., 1.])

source_ha = make_volumesource(
    field_name='ha', ds=ds, lower=5e-18, upper=1e-15,
    cmap='Blues_r', scale=1., log=True, nlayers=8,
    scale_func=linramp
)
sc.add_source(source_ha)

source_hp = make_volumesource(
    field_name='hp', ds=ds, lower=5e-18, upper=1e-15,
    cmap='myRed', scale=1., log=True, nlayers=8,
    scale_func=linramp
)
sc.add_source(source_hp)

source_h2 = make_volumesource(
    field_name='h2', ds=ds, lower=5e-22, upper=1e-15,
    cmap='dusk', scale=1., log=True, nlayers=8,
    scale_func=quadramp
)
sc.add_source(source_h2)

#point_colors = np.ones((sink_pos.shape[0], 4))
#point_colors[:, 3] = 0.3
#point_radii = np.ones(sink_pos.shape[0], dtype=int)
#point_radii *= 1
#points = PointSource(sink_pos * cm, colors=point_colors, radii=point_radii)
#sc.add_source(points)

cam = sc.add_camera()
#cam.position = [1, 1, 0.1]
#cam.position = [0.3, 0.0, 0.05]
cam.position = [1.2, 0.3, 0.0]
#cam.focus = ds.domain_center
cam.focus = [0.0, 0.0, 0.0]
#cam.north_vector = [0.0, 1.0, 0.0]
#cam.width = [.3, .3, 0.3]
cam.switch_orientation(north_vector=[0, 1, 0])
cam.resolution = np.asarray([1024, 1024], dtype=int) // 2

sc._sigma_clip = 10
im = sc.render()


scale_length = 100  # in pc
xl = 20
xr = xl + float(im.shape[0]) * scale_length / ((xmax - xmin)[0] / const.parsec * const.centi)

fig = plt.figure(figsize=(3,3))
ax = fig.gca()
ax.set_position([0, 0, 1, 1])

if prefac:
    im_boost = im.copy()
    im_boost = np.transpose(im_boost, axes=(1, 0, 2))
    im_boost[:, :, :3] *=  1./im_boost[:, :, :3].max()
    ax.imshow(im_boost)
else:
    ax.imshow(np.transpose(im, axes=(1, 0, 2)))
ax.axis('off')
#ax.text(
#    5, 5,
#    r'%s' % 'RSN-4C'
#    '\n'
#    r't = %2.2f Myr' % time,
#    color='w', verticalalignment='top'
#)

#ax.plot(np.asarray([xl, xr]), np.asarray([im.shape[1]-10, im.shape[1]-10]), c='w')
#ax.text(
#    (xl + xr) / 2., im.shape[1] - 10 - 2,
#    '%i pc' % scale_length,
#    color='w', verticalalignment='bottom',
#    horizontalalignment='center'
#)

fig.axes[0].get_xaxis().set_visible(False)
fig.axes[0].get_yaxis().set_visible(False)
fig.savefig(
    #fname="volr_full_%04d%s.pdf" % (int(sys.argv[1]), add_name),
    fname='volr_full.pdf',
    format='pdf', dpi=300,
    bbox_inches='tight', pad_inches=0.
)

# sc.save("volr_full.png")
