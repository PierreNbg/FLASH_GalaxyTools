import numpy as np
from scipy.interpolate import RegularGridInterpolator


def init():
    import scipy.constants as const
    import flash_amr_tools as amr_tools
    import h5py
    
    # fname = '../../../simulation_snapshots/DwarfGal_hdf5_chk_0000'
    fname = 'DwarfGal_hdf5_chk_0000'
    blist, brefs, bns = amr_tools.get_true_blocks(fname, xmin=[], xmax=[])
    pf = h5py.File(fname, 'r')
    
    dtype = np.float64
    
    bbox = pf["bounding box"][()].astype(dtype)
    gid = pf['gid'][()]
    ref_lvl = pf['refine level'][()]
    bs = pf['block size'][()].astype(dtype)
    
    dens = pf["dens"][()].astype(dtype)
    acx = pf['gacx'][()].astype(dtype)
    acy = pf['gacy'][()].astype(dtype)
    acz = pf['gacz'][()].astype(dtype)
    
    p0 = 1e-18
    
    pres, bt = integrate_amr(
        dens, acc=[acx, acy, acz],
        bbox=bbox, gid=gid, blist=blist, p0=p0, rlvl=ref_lvl,
        bs=bs, start_bid=blist[0]
    )
    pres, bt = integrate_amr(
        dens=np.ones_like(dens),
        acc=[np.ones_like(acx), np.ones_like(acx), np.ones_like(acx)],
        bbox=bbox, gid=gid, blist=blist, p0=p0, bs=bs, start_bid=blist[0]
    )
    
    pres /= 8e3 / 2048 * const.parsec / const.centi


# Create a padding for datasets
def create_ghostcells(data):
    data_pad = np.pad(
        data,
        pad_width=(
            (0, 0),
            (1, 1),
            (1, 1),
            (1, 1)
        )
    )
    return data_pad

# Create bounding box which includes the GC
def create_bbox_gc(bbox, bs, bshape=(8, 8, 8)):
    # Determine cell size
    dx = bs / bshape
    # Update bbox to account for gc
    bbox_gc = bbox.copy()
    bbox_gc[:, :, 0] -= dx
    bbox_gc[:, :, 1] += dx
    
    return bbox_gc

# Determine the cell centers of all cell within a block
def get_cell_centers(bbox, bshape=(8, 8, 8)):
    # Determine the number of blocks handed in
    nr_blocks = bbox.shape[0]
    
    # Create empty array to store the cell centers
    arr_shape = tuple([nr_blocks] + list(bshape) + [3])
    ccoord = np.zeros(arr_shape)
    
    # Loop over all blocks
    for i in range(nr_blocks):
        # Determine cell size
        dx = (bbox[i, :, 1] - bbox[i, :, 0]) / bshape
        # Determine linear spacing of cells in x-, y- and z-direction
        # accounting for a shift of half a cell size to go to cell center from
        # bounding box data
        x = np.linspace(
            bbox[i, 0, 0], bbox[i, 0, 1], bshape[0], endpoint=False
        ) + dx[0] / 2.
        y = np.linspace(
            bbox[i, 1, 0], bbox[i, 1, 1], bshape[1], endpoint=False
        ) + dx[1] / 2.
        z = np.linspace(
            bbox[i, 2, 0], bbox[i, 2, 1], bshape[2], endpoint=False
        ) + dx[2] / 2.
        # Create meshgrid to get all cell positions
        zz, yy, xx = np.meshgrid(z, y, x, indexing='ij')

        # Store data
        ccoord[i, ..., 0] = zz
        ccoord[i, ..., 1] = yy
        ccoord[i, ..., 2] = xx
    
    return ccoord

# Interpolate data of a block for a given set of positions
# data_blk  - Data used for interpolation
# bbox_blk  - Bounding box of the block
#             Used for determining cell centers of the grid
# points    - Points which needs to be determined
#             Needs to be of shape:
#              - (Ni, 3)
#              - (Ni, Nj, 3)
#              - (Ni, Nj, Nk, 3) ...
def interpolate_data_3d(
    data_blk, bbox_blk, points,
    method='linear',
    debug=False
):
    # Get block shape
    bshape = data_blk.shape
    # Determine corresponding block size
    dx = (bbox_blk[:, 1] - bbox_blk[:, 0]) / bshape[::-1]
    # Create linear spacing of cell centers
    x = np.linspace(
        bbox_blk[0, 0], bbox_blk[0, 1], bshape[2], endpoint=False
    ) + 0.5 * dx[0]
    y = np.linspace(
        bbox_blk[1, 0], bbox_blk[1, 1], bshape[1], endpoint=False
    ) + 0.5 * dx[1]
    z = np.linspace(
        bbox_blk[2, 0], bbox_blk[2, 1], bshape[0], endpoint=False
    ) + 0.5 * dx[2]

    if debug:
        print("interpolate_data")
        print("\tx", x)
        print("\ty", y)
        print("\tz", z)

    # Create interpolator
    f = RegularGridInterpolator(
        points=(z, y, x),
        values=data_blk,
        method=method,
        bounds_error=False,
        fill_value=None
    )
    
    return f(points)


def interpolate_data_2d(
        data_blk, bbox_blk, points,
        method='linear', axis=(0, 1),
        debug=False
):
    # Get block shape
    bshape = data_blk.shape
    # Determine corresponding block size
    dx = (bbox_blk[:, 1] - bbox_blk[:, 0]) / bshape[::-1]
    # Create linear spacing of cell centers
    x = np.linspace(
        bbox_blk[axis[0], 0],
        bbox_blk[axis[0], 1],
        bshape[2-axis[0]],
        endpoint=False
    ) + 0.5 * dx[axis[0]]
    y = np.linspace(
        bbox_blk[axis[1], 0],
        bbox_blk[axis[1], 1],
        bshape[2-axis[1]],
        endpoint=False
    ) + 0.5 * dx[axis[1]]

    sl_cells = [slice(None), slice(None), slice(None), np.asarray(axis)]

    if debug:
        print("interpolate_data")
        print("\tx", x.shape)
        print("\ty", y.shape)
        print("\tx", x)
        print("\ty", y)
        print("\tdata", np.squeeze(data_blk).shape)
        print("\tdata", data_blk.min(), data_blk.max())
        print("\taxis", axis)
        print("\tslices", sl_cells)
        print("\tpoints", points.shape)

    points = np.squeeze(points[*sl_cells])

    # Create interpolator
    f = RegularGridInterpolator(
        points=(y, x),
        values=np.squeeze(data_blk),
        method=method,
        bounds_error=False,
        fill_value=None
    )
    
    
    return f(np.squeeze(points))


# Update GC to contain FACE data
def update_gc_face(
    data_gc, ddata_gc, bid,
    axis, direction,
    debug=False
):
    # Check that bid is a list with block ids
    if type(bid) == np.ndarray:
        bid = bid.copy().tolist()
    elif type(bid) is not list:
        bid = [bid]
        
    sl = [slice(1, 9), slice(1, 9), slice(1, 9)]
    
    sl_prev = sl.copy()
    sl_gc = sl.copy()
    
    if direction > 0:
        sl_prev[2 - axis] = slice(8, 9)
        sl_gc[2 - axis] = slice(9, 10)
    else:
        sl_prev[2 - axis] = slice(1, 2)
        sl_gc[2 - axis] = slice(0, 1)
    
    # Copy data from previous cell
    data_gc[bid, *sl_gc] = data_gc[bid, *sl_prev].copy()
    # Add half step to reach the face
    data_gc[bid, *sl_gc] += ddata_gc[bid, *sl_prev].copy()
    
    return data_gc


# Wrapper to update guard cells in both directions
def fill_gc(
    data_gc, bid,
    gid, bbox, bs, rlvl,
    axis,
    use_face=False,
    debug=False
):

    for direction in [-1, 1]:
        # Update all guard cells in given direction
        data_gc = fill_gc_blks(
            data_gc=data_gc,
            bid=bid, gid=gid,
            bbox=bbox, bs=bs, rlvl=rlvl,
            axis=axis, direction=direction,
            use_face=use_face,
            debug=debug
        )
        
    return data_gc


def get_neighbour_slice(cbid, nbid, bbox):
    # Determine overlap between neighbours
    # |  cbid   |  nbid   |
    # |         |         |
    # |         |---------|
    # |         |         |
    # |         |         |
    # Determine positions to query for neighbour
    # Determine offset of neighbour block to current block
    x0 = np.round(
        8 * (bbox[nbid, :, 0] - bbox[cbid, :, 0])
        / (bbox[cbid, :, 1] - bbox[cbid, :, 0])
    ).astype(int) + 1
    x1 = (np.round(
        8 * (bbox[nbid, :, 1] - bbox[cbid, :, 0])
        / (bbox[cbid, :, 1] - bbox[cbid, :, 0]))
    ).astype(int) + 1
    
    # Size of the neighbour nbid in relation
    # to the current block cbid
    # sub_size == 4     (neighbour is one level higher)
    # sub_size == 8     (neighbour is at same level)
    # sub_size == 16    (neighbour is one level lower)
    sub_size = (x1 - x0).max()
    
    # In case of the neighbour being larger than
    # the current block we need to limit the indices
    # to the range [1, 9]
    x0 = np.maximum(x0, 1)
    x1 = np.minimum(x1, 9)
    
    # Set of slices zyx order to get the correct cell centers
    sl_sub = [
        slice(x0[2], x1[2]),
        slice(x0[1], x1[1]),
        slice(x0[0], x1[0])
    ]
    
    return sl_sub, sub_size


def fill_gc_blks_face(
    data_gc, bid,
    gid, bbox, bs,
    axis, direction,
    debug=False
):
    
    # Check that bid is a list with block ids
    if type(bid) == np.ndarray:
       bid = bid.copy().tolist()
    elif type(bid) is not list:
       bid = [bid]
    
    # Get all neighbour block ids
    neigh_list = get_neighbour_bids(
        bid=bid, gid=gid,
        axis=axis, direction=direction,
        return_is_parent=False,
        no_children=False
    )

    # Determine bbox which includes GC
    bbox_gc = create_bbox_gc(bbox=bbox[bid], bs=bs[bid])
    # Determine cell centers including GC
    cell_cnt_gc = get_cell_centers(
        bbox=bbox_gc,
        bshape=(10, 10, 10)
    )

    # Loop over all blocks that need to be updated
    for i in range(len(bid)):
        cbid = bid[i]
        
        if debug:
            print("Current bid:", cbid)
            
        # Loop over all neighbours of that block
        # in given direction along given axis
        for j in range(len(neigh_list[i])):
            # Get block if of neighbour
            nbid = neigh_list[i][j]
            
            if debug:
                print("\tCurrent neighbour:", nbid)
            
            sl_sub_cur, sub_size_cur = get_neighbour_slice(cbid, nbid, bbox)
            sl_sub_nei, sub_size_nei = get_neighbour_slice(nbid, cbid, bbox)
            sl_gc = [slice(1, 9), slice(1, 9), slice(1, 9)]

            # Replace axis entry to get slice
            # which needs to be updated
            # and get the correct cell positions
            if direction > 0:
                sl_sub_cur[2 - axis] = slice(9, 10)
                sl_sub_nei[2 - axis] = slice(0, 1)
                sl_gc[2 - axis] = slice(0, 1)
            else:
                sl_sub_cur[2 - axis] = slice(0, 1)
                sl_sub_nei[2 - axis] = slice(9, 10)
                sl_gc[2 - axis] = slice(9, 10)
            
            if debug:
                print("Sub slices:", sl_sub_cur)
            
            # Get data from neighbour GCs
            # data_blk = data_gc[nbid, *sl_sub_nei].copy()
            data_blk = data_gc[nbid, *sl_gc].copy()
            
            dat_out = np.zeros_like(
                data_gc[cbid, *sl_sub_cur],
                dtype=data_gc.dtype
            )
            
            # Check if we need to average or duplicate data
            # due to level change
            if sub_size_cur == 8:
                # Same level, copy data as is
                data_blk = data_gc[nbid, *sl_sub_nei].copy()
                dat_out = data_blk
            elif sub_size_cur == 4:
                # TODO: Need to test interpolation here
                # Neighbour is smaller than current block
                # need to average face values
                # Remove axis which was already reduced
                # Remaining two indices are the remaining axis
                # where we need to average over
                sl_ind = [2, 1, 0]
                sl_ind.pop(axis)
                tmp_sl = [slice(None), slice(None), slice(None)]
                
                data_blk = data_gc[nbid, *sl_sub_nei].copy()
                
                tmp_sl[sl_ind[0]] = slice(0, None, 2)
                tmp_sl[sl_ind[1]] = slice(0, None, 2)
                dat_out += data_blk[*tmp_sl]
                
                tmp_sl[sl_ind[0]] = slice(1, None, 2)
                tmp_sl[sl_ind[1]] = slice(0, None, 2)
                dat_out += data_blk[*tmp_sl]

                tmp_sl[sl_ind[0]] = slice(0, None, 2)
                tmp_sl[sl_ind[1]] = slice(1, None, 2)
                dat_out += data_blk[*tmp_sl]
                
                tmp_sl[sl_ind[0]] = slice(1, None, 2)
                tmp_sl[sl_ind[1]] = slice(1, None, 2)
                dat_out += data_blk[*tmp_sl]
                
                dat_out /= 4.0
                
            elif sub_size_cur == 16:
                # TODO: Need to interpolate
                # Neighbour is larger than current block
                # Need to duplicate data to fit current shape
                
                # Determine the bounding box of the face data
                bbox_blk = bbox[nbid].copy()
                if direction > 0:
                    bbox_blk[axis, 1] = bbox_blk[axis, 0]
                else:
                    bbox_blk[axis, 0] = bbox_blk[axis, 1]
                
                # Select cell centers of GC
                points = cell_cnt_gc[i, *sl_sub_cur].copy()
                dx_blk_cur = bs[cbid] / 8.
                # Shift the position to FACE along the current axis
                if direction > 0:
                    points[..., 2 - axis] -= dx_blk_cur[axis] / 2.
                else:
                    points[..., 2 - axis] += dx_blk_cur[axis] / 2.
                
                # Interpolate points of the current block
                # using the values of the neighbour FACE data
                dat_out = interpolate_data_3d(
                    data_blk=data_blk,
                    bbox_blk=bbox_blk,
                    points=points,
                )
                
                #sl_ind = [2, 1, 0]
                #sl_ind.pop(axis)
                
                #data_blk = np.repeat(data_blk, axis=sl_ind[0], repeats=2)
                #data_blk = np.repeat(data_blk, axis=sl_ind[1], repeats=2)
                #dat_out += data_blk
            else:
                dat_out = 0.0
                print("ERROR: Unable to determine block level difference")

            # Update GC of block
            data_gc[cbid, *sl_sub_cur] = dat_out
    
    return data_gc

# Interpolate values for filling guard cells
# Only needed when neighbour block level differs
# from current block
# data_gc   - data set including GC
# bid       - list of block ids to update
# gid       - global id list
# bbox      - bounding box of each block (without GC)
# bs        - block size (without GC)
# axis      - axis along to update GC
# direction - direction where we update GC
def fill_gc_blks(
    data_gc, bid,
    gid, bbox, bs, rlvl,
    axis, direction,
    use_face=False,
    debug=False,
):
    # Check that bid is a list with block ids
    if type(bid) == np.ndarray:
       bid = bid.copy().tolist()
    elif type(bid) is not list:
       bid = [bid]

    # Get all neighbour block ids
    neigh_list = get_neighbour_bids(
        bid=bid, gid=gid,
        axis=axis, direction=direction,
        return_is_parent=False,
        no_children=False
    )
    
    # Determine slice of data points that
    # need to be updated
    sl_cur = [slice(1, 9), slice(1, 9), slice(1, 9)]
    if direction > 0:
        sl_cur[2 - axis] = slice(9, 10)
    else:
        sl_cur[2 - axis] = slice(0, 1)
    
    # Slices to cut GCs
    sl_dat = [slice(1, 9), slice(1, 9), slice(1, 9)]
    
    # Determine bbox which includes GC
    bbox_gc = create_bbox_gc(bbox=bbox[bid], bs=bs[bid])
    # Determine cell centers including GC
    cell_cnt_gc = get_cell_centers(
        bbox=bbox_gc,
        bshape=(10, 10, 10)
    )
    
    if debug:
        print("bbox_gc:", bbox_gc.shape)
        print("cell_cnt_gc:", cell_cnt_gc.shape)
    
    # Loop over all blocks that need to be updated
    for i in range(len(bid)):
        cbid = bid[i]

        if debug:
            print("Current bid:", cbid)
            
        # Loop over all neighbours of that block
        # in given direction along given axis
        for j in range(len(neigh_list[i])):
            # Get block if of neighbour
            nbid = neigh_list[i][j]
            
            if debug:
                print("\tCurrent neighbour:", nbid)
            
            # Determine positions to query for neighbour
            # Determine offset of neighbour block to current block
            x0 = np.round(
                8 * (bbox[nbid, :, 0] - bbox[cbid, :, 0])
                / (bbox[cbid, :, 1] - bbox[cbid, :, 0])
            ).astype(int) + 1
            x1 = (np.round(
                8 * (bbox[nbid, :, 1] - bbox[cbid, :, 0])
                / (bbox[cbid, :, 1] - bbox[cbid, :, 0]))
                 ).astype(int) + 1
            
            # In case of the neighbour being larger than
            # the current block we need to limit the indices
            # to the range [1, 9]
            x0 = np.maximum(x0, 1)
            x1 = np.minimum(x1, 9)
            
            # Set of slices zyx order to get the correct cell centers
            sl_sub = [
                slice(x0[2], x1[2]),
                slice(x0[1], x1[1]),
                slice(x0[0], x1[0])
            ]
            
            # Replace axis entry to get slice
            # which needs to be updated
            # and get the correct cell positions
            if direction > 0:
                sl_sub[2 - axis] = slice(9, 10)
            else:
                sl_sub[2 - axis] = slice(0, 1)
            
            if debug:
                print("Sub slices:", sl_sub)
            
            # Get the cell centers for which we need to
            # interpolate values, sub-selected to points
            # overlapping with current neighbour block
            points = cell_cnt_gc[i, *sl_sub]
            
            # When we want to determine the face values
            # we need to shift position to bounding box
            # positions
            if use_face:
                tmp_dx = bs[cbid] / 8.
                if direction > 0:
                    points[..., 2 - axis] -= 0.5 * tmp_dx[axis]
                else:
                    points[..., 2 - axis] += 0.5 * tmp_dx[axis]
            
            # Get data from neighbour without GC
            data_blk = data_gc[nbid, *sl_dat]
            
            if use_face:
                sl_gc = sl_dat.copy()
                if direction > 0:
                    sl_gc[2 - axis] = slice(0, 1)
                else:
                    sl_gc[2 - axis] = slice(9, 10)
                data_blk = data_gc[nbid, *sl_gc]
            
            # Get bbox of neighbour
            bbox_blk = bbox[nbid].copy()
            
            if use_face:
                if direction > 0:
                    bbox_blk[axis, 1] = bbox_blk[axis, 0]
                else:
                    bbox_blk[axis, 0] = bbox_blk[axis, 1]
            
            if debug:
                print("Points:", points.shape)
                print("Points:", points)
            
            method = "nearest"
            if rlvl[cbid] != rlvl[nbid]:
                method = 'linear'
                #if rlvl[cbid] < rlvl[nbid]:
                #    data_blk = np.log10(data_blk)
                #    data_blk[np.isinf(data_blk)] = 0.0
            
            dat_out = interpolate_data_3d(
                data_blk=data_blk,
                bbox_blk=bbox_blk,
                points=points,
                method=method
            )
        
            #if method != "nearest" and rlvl[cbid] < rlvl[nbid]:
            #    dat_out = 10**dat_out
                
            # Update GC of block
            data_gc[cbid, *sl_sub] = dat_out

    return data_gc


# Determines block ids of neighbours
# bid       - current block id
# gid       - global list of block ids
# axis      - axis along which to check
#             0 -- x
#             1 -- y
#             2 -- z
# direction - positive or negative direction
#               1      or   -1
def get_neighbour_bids(
    bid, gid,
    axis, direction,
    return_is_parent=False,
    no_children=False
):
    # Get the entry within gid which gives the
    # neighbour block
    gid_sel = axis * 2 + (1 + direction) // 2
    
    # Get correct children bids of neighbour
    # if neighbour has children
    child_sel = np.arange(8)
    child_sel = child_sel[(child_sel // 2**axis + (1 - direction) // 2) % 2 == 0]
    
    # Check that bid is a list with block ids
    if type(bid) == np.ndarray:
       bid = bid.copy().tolist()
    elif type(bid) is not list:
       bid = [bid]
    
    neigh_list = []
    is_parent = []
    
    # Loop over all block ids
    for it in bid:
        # Include offset for Fortran indexing
        tmp_neigh = gid[it, gid_sel].tolist() - 1
        # If the neighbour exist add it
        if tmp_neigh >= 0:
            neigh_list.append([tmp_neigh])
            is_parent.append(False)
            # If the neighbour has children update
            # the list with the children ids
            if gid[tmp_neigh, -1] >= 0 and not no_children:
                # Select only children which have an interface
                # with the block in the current direction
                # Account for Fortran indexing
                tmp_neigh = (gid[tmp_neigh, 7:][child_sel] - 1).tolist()
                neigh_list[-1] = tmp_neigh
        # Either at domain edge
        # or no neighbour at current level (take parent instead)
        elif tmp_neigh == -1 - 1:
            tmp_par = gid[it][6] - 1
            tmp_neigh = gid[tmp_par, gid_sel].tolist() - 1
            neigh_list.append([tmp_neigh])
            is_parent.append(True)
        else:
            neigh_list.append([])
            is_parent.append(False)


    if return_is_parent:
        return neigh_list, is_parent
    else:
        return neigh_list

def integrate_blk(
    pres_gc, pgrad_gc,
    sl, axis, direction=1,
    use_face=False,
    debug=False
):
    
    sl_cur = sl.copy()
    sl_prev = sl_cur.copy()
    
    sl_start = sl[2 - axis].start
    sl_stop = sl[2 - axis].stop
    
    sl_prev[2 - axis] = slice(
        sl_start - direction,
        sl_stop - direction,
    )

    if debug:
        print("integrate_blk")

    nloops = 8
    if use_face:
        nloops = 9
    
    if debug:
        print("\tnloops:", nloops)
    
    # Loop over all cells in block up to GC
    for i in range(nloops):
        if debug:
            print("\t\tsl_prev:", sl_prev)
            print("\t\tsl_cur:", sl_cur)
        
        # Copy value of previous cell
        pres_gc[:, *sl_cur] = pres_gc[:, *sl_prev].copy()
        
        # Add half step of previous cell
        if not use_face:
            pres_gc[:, *sl_cur] += pgrad_gc[axis][:, *sl_prev]
        elif sl_start - direction != 0 and sl_start - direction != 9:
            pres_gc[:, *sl_cur] += pgrad_gc[axis][:, *sl_prev]
        
        # Add half step of current cell
        if not use_face:
            pres_gc[:, *sl_cur] += pgrad_gc[axis][:, *sl_cur]
        elif sl_start != 0 and sl_start != 9:
            pres_gc[:, *sl_cur] += pgrad_gc[axis][:, *sl_cur]
        
        # Update previous slice
        sl_prev = sl_cur.copy()
        
        sl_cur[2 - axis] = slice(
            sl_start + direction,
            sl_stop + direction,
            )
        
        # Update current slice
        sl_start = sl_cur[2 - axis].start
        sl_stop = sl_cur[2 - axis].stop

    
    return pres_gc


def integrate_amr(
    dens, acc, bbox, gid, bs, rlvl,
    blist, p0, start_bid,
    debug=False,
    update_par=False
):
    
    # Check that bid is a list with block ids
    if type(blist) == np.ndarray:
       blist = blist.copy().tolist()
    elif type(blist) is not list:
       blist = [blist]
    
    # Get cell size for each block
    dx = (bbox[:, :, 1] - bbox[:, :, 0]) / dens.shape[1:]

    print("Creating GC")
    # Create pressure gradient and add ghost cells for data sets
    pgradx_gc = create_ghostcells(
        data=dens * acc[0] * dx[:, 0, None, None, None] * 0.5
    )
    pgrady_gc = create_ghostcells(
        data=dens * acc[1] * dx[:, 1, None, None, None] * 0.5
    )
    pgradz_gc = create_ghostcells(
        data=dens * acc[2] * dx[:, 2, None, None, None] * 0.5
    )

    print('Filling GC')
    # Fill ghost cells with data of neighbouring blocks
    # Only need to update GC for the corresponding directions
    #if False:
    #    pgradx_gc = fill_gc(
    #        data_gc=pgradx_gc, bid=blist,
    #        gid=gid, bbox=bbox, bs=bs, rlvl=rlvl,
    #        axis=0
    #    )
    #    pgrady_gc = fill_gc(
    #        data_gc=pgrady_gc, bid=blist,
    #        gid=gid, bbox=bbox, bs=bs, rlvl=rlvl,
    #        axis=1
    #    )
    #    pgradz_gc = fill_gc(
    #        data_gc=pgradz_gc, bid=blist,
    #        gid=gid, bbox=bbox, bs=bs, rlvl=rlvl,
    #        axis=2
    #    )
    
    if update_par:
        print('Updating parents')
        pgradx_gc[:, 1:9, 1:9, 1:9] = update_all_parent_blocks(
            data=pgradx_gc[:, 1:9, 1:9, 1:9],
            gid=gid,
            sel=blist
        )
        pgrady_gc[:, 1:9, 1:9, 1:9] = update_all_parent_blocks(
            data=pgrady_gc[:, 1:9, 1:9, 1:9],
            gid=gid,
            sel=blist
        )
        pgradz_gc[:, 1:9, 1:9, 1:9] = update_all_parent_blocks(
            data=pgradz_gc[:, 1:9, 1:9, 1:9],
            gid=gid,
            sel=blist
        )
        
    # Output array
    pres_gc = np.zeros_like(pgradx_gc, dtype=dens.dtype)
    blist = sorted(blist)
    
    print('Integrate on AMR grid')
    
    # Get starting block
    if start_bid is None:
        start_bid = blist[0]
        print('\tFirst block:', start_bid)
    
    # Set initial pressure in starting block
    pres_gc[start_bid, 1, 1, 0] = p0
    pres_gc[start_bid, 1, 0, 1] = p0
    pres_gc[start_bid, 0, 1, 1] = p0
    
    # Create initial slice
    sl = [
        slice(1, 2, None),
        slice(1, 2, None),
        slice(1, 2, None)
    ]
    
    # Axis to integrate along
    cax = 0
    # Direction to integrate (positive or negative direction
    cdir = 1
    
    cur_bid = [start_bid]
    blocks_touch = []
    
    blocks_time = []
    
    # Loop over all axis
    for i in range(3):
        print("\tIntegrating along axis", i)
        # Reset exit condition
        at_domain_edge = False
        first = True
        
        if i == 0:
            pass
            
        # Update slice and blocks which we start from
        if i == 1:
            blocks_touch = np.unique(blocks_touch).tolist()
            cur_bid = blocks_touch.copy()
            sl[2 - cax] = slice(1, 9, None)
            cax = 1
            # Don't reuse boundary value for first integration step
            # in new direction as these have been filled in the
            # previous iteration
            sl[2 - cax] = slice(2, 3, None)
        
        if i == 2:
            blocks_touch = np.unique(blocks_touch).tolist()
            cur_bid = blocks_touch.copy()
            sl[2 - cax] = slice(1, 9, None)
            cax = 2
            # Don't reuse boundary value for first integration step
            # in new direction as these have been filled in the
            # previous iteration
            sl[2 - cax] = slice(2, 3, None)
        
        # Loop as long as we don't have reached the end
        # of the domain for all blocks
        while not at_domain_edge:
            if debug:
                print('\tBlock id', cur_bid)
                
            # Add all blocks to list for sanity check at the end
            blocks_time.append(cur_bid)
            blocks_touch += cur_bid
            
            if cax == 2:
                # Get list of previous neighbours
                neigh_bid = get_neighbour_bids(
                    bid=cur_bid, gid=gid,
                    axis=cax, direction=-1*cdir
                )
                neigh_bid = sorted([it for cbid in neigh_bid for it in cbid])
                # Update the GC of the previous neighbours
                # to contain face values
                pres_gc = update_gc_face(
                    data_gc=pres_gc, ddata_gc=pgradz_gc,
                    bid=neigh_bid,
                    axis=cax, direction=cdir,
                    debug=False
                )

            # Update ghost cells of current blocks
            # with data from neighbours
            pres_gc = fill_gc(
                data_gc=pres_gc, bid=cur_bid,
                gid=gid, bbox=bbox, bs=bs, rlvl=rlvl,
                axis=cax,
                use_face=True,
                debug=debug
            )
            
            # Get pressure gradients to integrate
            pgrad_gc = [
                pgradx_gc[cur_bid],
                pgrady_gc[cur_bid],
                pgradz_gc[cur_bid]
            ]

            # Integrate data in block along given axis
            # GC will contain FACE values
            # which need to be communicated to neighbours
            pres_gc[cur_bid] = integrate_blk(
                pres_gc=pres_gc[cur_bid], pgrad_gc=pgrad_gc,
                sl=sl, axis=cax, direction=cdir,
                debug=debug, use_face=True
            )
            
            if update_par:
                # Update data of all parents of current blocks
                # before determining neighbours
                pres_gc[:, 1:9, 1:9, 1:9] = update_all_parent_blocks(
                    data=pres_gc[:, 1:9, 1:9, 1:9],
                    gid=gid,
                    sel=cur_bid
                )
            
            
            # Get new list of neighbours from current blocks
            cur_bid = get_neighbour_bids(
                bid=cur_bid, gid=gid,
                axis=cax, direction=cdir,
                return_is_parent=False,
                no_children=False
            )
            # Flatten list to 1D
            cur_bid = sorted([it for cbid in cur_bid for it in cbid])
            
            # Reset integration to include first value
            # GC should contain correct value now
            sl[2 - cax] = slice(1, 2, None)
            
            # If no more neighbours exist we must be at the edge of the domain
            if len(cur_bid) == 0:
                at_domain_edge = True
    
    print('\tBlocks touched:', np.unique(blocks_touch).size)
    
    return pres_gc, blocks_time


# Update all parents with data from their children
def update_all_parent_blocks(data, gid, sel=None):
    # Get list of block ids
    # Going backwards through the list
    # will automatically fill all parents
    blist = np.arange(gid.shape[0])[::-1]
    
    if sel is not None:
        blist = sel
        if type(blist) is not list:
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
