import numpy as np
import os
import yt
import sys

from yt.visualization.volume_rendering.transfer_function_helper import TransferFunctionHelper
from yt.visualization.volume_rendering.api import Scene
from yt.visualization.volume_rendering.render_source import KDTreeVolumeSource as VolumeSource
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
from matplotlib import colormaps


# alpha functions
def constval_atom(vals, minval, maxval):
    return 0.1

def constval(vals, minval, maxval):
    return 0.5

def sqtramp(vals, minval, maxval):
    return np.sqrt(vals - vals.min())/np.sqrt(vals.max() - vals.min())

def linramp(vals, minval, maxval):
    return (vals - vals.min())/(vals.max() - vals.min())

def linramp_half(vals, minval, maxval):
    return 0.5*(vals - vals.min())/(vals.max() - vals.min())

def quadramp(vals, minval, maxval):
    return np.abs(vals - vals.min())**2/np.abs(vals.max() - vals.min())**2

def sigmoid(vals, minval, maxval):
    #lower = 1e-22
    #upper = 5e-16
    #print(vals)
    return 1./(1. + np.exp(-8*(vals-(vals.max()*0.7-vals.min()*0.3))))

def create_source(ds, field, log, opacity, bounds, cmap, scale_func, name):
    vs = VolumeSource(ds, field=field)
    vs.use_ghost_zones = True

    tfh = TransferFunctionHelper(ds)
    tfh.set_field(field)
    tfh.set_log(log)
    tfh.grey_opacity = opacity
    tfh.set_bounds(bounds)
    tfh.build_transfer_function()

    lb, ub = bounds
    if log:
        lb = np.log10(lb)
        ub = np.log10(ub)
        
    tfh.tf.map_to_colormap(
        lb, ub,
        colormap=cmap,
        scale_func=scale_func
    )
    
    tfh.plot(name, profile_field=field)
    vs.set_transfer_function(tfh.tf)
    
    return vs


class pltSource:
    def __init__(self, prefix, saveDir, targetDir, key):
        self.prefix = prefix
        self.saveDir = saveDir
        self.targetDir = targetDir
        self.key = key


pltSrc = pltSource(
    "rfl10_update",
    "./pics_vol_ren/",
    "./../rfl10_update/",
    "SpitzerTest_hdf5_plt_cnt"
)

prefix = pltSrc.prefix
saveDir = pltSrc.saveDir
targetDir = pltSrc.targetDir

allItems = os.listdir(targetDir)
allItems = [k for k in allItems if pltSrc.key in k]
allItems = sorted(allItems)

final_arr = []
final_arr.append( np.array([0.3,0.0,0.0]) )
final_arr.append( np.array([0.0,1.0,0.0]) )
final_arr.append( np.array([0.,0.,0.]) )


init_arr = []
init_arr.append( np.array([0.20,0.40,0.2])*0.3 )
init_arr.append( np.array([0.0,0.0,0.5]) )
init_arr.append( np.array([0.,0.,0.]) )
render_arr = final_arr

targetDir = './'
prefix = './'
saveDir = './'
fname = 'SpitzerTest_hdf5_plt_cnt_0230'
item = fname

for item in allItems:
    counter = int(item[-4:])
    ds = yt.load(targetDir + item)

    print("%s_rendering_%04d.png"%(prefix,int(counter)))

    #create custom colormap
    blue_cm = LinearSegmentedColormap.from_list("myBlue", [(0,0,1), (0,0,1)], N=2)
    green_cm = LinearSegmentedColormap.from_list("myGreen", [(0,1,0), (0,1,0)], N=2)
    yell_cm = LinearSegmentedColormap.from_list("myYell", [(1,1,0), (1,1,0)], N=2)
    red_cm = LinearSegmentedColormap.from_list("myRed", [(1,0,0), (1,0,0)], N=2)
    dens_cm = LinearSegmentedColormap.from_list("myDens", [(1,1,1), (1,1,1)], N=2)

    colormaps.register(cmap=blue_cm)
    colormaps.register(cmap=green_cm)
    colormaps.register(cmap=yell_cm)
    colormaps.register(cmap=red_cm)
    colormaps.register(cmap=dens_cm)
    

    lower_h2 = 5e-22
    upper_h2 = 5e-16
    bounds_h2 = (lower_h2, upper_h2)

    lower_ha = 5e-21
    upper_ha = 2e-13
    bounds_ha = (lower_h2, upper_h2)

    lower_hp = 5e-21
    upper_hp = 2e-13
    bounds_hp = (lower_hp, upper_hp)

    sc = Scene()
    sc.background_color = np.array([1.,1.,1.,1.])

    # set up camera
    cam = sc.add_camera(ds, lens_type='perspective')
    res =  512 #1024 #4096
    cam.resolution = [res, res]
    cam.position = render_arr[0]
    cam.north_vector= render_arr[1]
    cam.focus = render_arr[2]

    def _ionized(field, data):
        return data["dens"] * data["ihp "]
    def _atomic(field, data):
        return data["dens"] * data["iha "]
    def _mole(field, data):
        return data["dens"] * data["ih2 "]

    ds.add_field(("gas", "ioni"), function=_ionized, units="g/cm**3", sampling_type="cell")
    ds.add_field(("gas", "atom"), function=_atomic, units="g/cm**3", sampling_type="cell")
    ds.add_field(("gas", "mole"), function=_mole, units="g/cm**3", sampling_type="cell")
    
    ds.force_periodicity()
    greyOpa = False
    cmap = cm.get_cmap('dusk')
    
    vs = create_source(
        ds=ds,
        field='mole',
        log=True, opacity=greyOpa,
        bounds=bounds_h2,
        cmap='dusk',
        scale_func=quadramp,
        name=saveDir + 'transfer_function_mole_%04d.png' % int(counter)
    )
    sc.add_source(vs)

    vs = create_source(
        ds=ds,
        field='ioni',
        log=True, opacity=greyOpa,
        bounds=bounds_hp,
        cmap='myRed',
        scale_func=linramp,
        name=saveDir + 'transfer_function_ioni_%04d.png' % int(counter)
    )
    sc.add_source(vs)
    
    vs = create_source(
        ds=ds,
        field='atom',
        log=True, opacity=greyOpa,
        bounds=bounds_ha,
        cmap='myBlue',
        scale_func=linramp,
        name=saveDir + 'transfer_function_atom_%04d.png' % int(counter)
    )
    sc.add_source(vs)
    
    sc.save(saveDir + "%s_rendering_%04d_z.pdf" % (prefix, int(counter)), sigma_clip=4)
