import numpy as np
import h5py
import flash_amr_tools as amr_tools

from io_amr import update_all_parent_blocks


def init():
    fname = 'DwarfGal_hdf5_chk_0000'
    blist, brefs, bns = amr_tools.get_true_blocks(fname, xmin=[], xmax=[])
    pf = h5py.File(fname, 'r')

    bbox = pf["bounding box"][()]
    gid = pf['gid'][()]
    ref_lvl = pf['refine level'][()]
    
    dens = pf["dens"][()]
    acx = pf['gacx'][()]
    acy = pf['gacy'][()]
    acz = pf['gacz'][()]
    
    p0 = 0.0
    
    pres = integrate_amr(dens, acc=[acx, acy, acz], bbox=bbox, gid=gid, blist=blist, p0=p0)


def get_cell_centers(bbox):
    nr_blocks = bbox.shape[0]
    
    ccoord = np.zeros((nr_blocks, 8, 8, 8, 3))
    
    for i in range(nr_blocks):
        dx = (bbox[i, :, 1] - bbox[i, :, 0]) / (8, 8, 8)
        x = np.linspace(bbox[i, 0, 0], bbox[i, 0, 1], 8, endpoint=False) + dx[0] / 2.
        y = np.linspace(bbox[i, 1, 0], bbox[i, 1, 1], 8, endpoint=False) + dx[1] / 2.
        z = np.linspace(bbox[i, 2, 0], bbox[i, 2, 1], 8, endpoint=False) + dx[2] / 2.
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        ccoord[i, ..., 0] = xx.T
        ccoord[i, ..., 1] = yy.T
        ccoord[i, ..., 2] = zz.T

    return ccoord

def integrate_amr(dens, acc, bbox, gid, blist, p0, start_bid=None, debug=False):
    if type(blist) is np.ndarray:
        blist = blist.tolist()
    
    # Get cell size for each block
    dx = (bbox[:, :, 1] - bbox[:, :, 0]) / dens.shape[1:]
    
    # Create pressure gradient and add ghost cells for data sets
    pgradx_gc = create_ghostcells(data=dens*acc[0]*dx[:, 0, None, None, None]*0.5)
    pgrady_gc = create_ghostcells(data=dens*acc[1]*dx[:, 1, None, None, None]*0.5)
    pgradz_gc = create_ghostcells(data=dens*acc[2]*dx[:, 2, None, None, None]*0.5)
    
    pgradx_gc = fill_ghostcells(data_pad=pgradx_gc, gid=gid, bbox=bbox, debug=debug, sel=blist)
    pgrady_gc = fill_ghostcells(data_pad=pgrady_gc, gid=gid, bbox=bbox, debug=debug, sel=blist)
    pgradz_gc = fill_ghostcells(data_pad=pgradz_gc, gid=gid, bbox=bbox, debug=debug, sel=blist)
    
    # Output array
    pres_gc = np.zeros_like(pgradx_gc)
    
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
    
    par_norm = 4.0
    # Loop over all axis
    for i in range(3):
        # Reset exit condition
        at_domain_edge = False
        
        if i == 0:
            par_norm = 1.0
        
        # Update slice and blocks which we start from
        if i == 1:
            blocks_touch = np.unique(blocks_touch).tolist()
            cur_bid = blocks_touch.copy()
            sl[2 - cax] = slice(1, 9, None)
            cax = 1
            # Don't reuse boundary value for first integration step
            # in new direction
            sl[2 - cax] = slice(2, 3, None)
            par_norm = 2.0
        
        if i == 2:
            blocks_touch = np.unique(blocks_touch).tolist()
            cur_bid = blocks_touch.copy()
            sl[2 - cax] = slice(1, 9, None)
            cax = 2
            # Don't reuse boundary value for first integration step
            # in new direction
            sl[2 - cax] = slice(2, 3, None)
            par_norm = 4.0
            
        # Loop as long as we don't have reached the end of the domain for all blocks
        while not at_domain_edge:
            if debug:
                print('\tBlock id', cur_bid)
                
            # Keep blocks back whose GC are not full updated yet
            # In case not all previous blocks have been able to update their values
            # to current blocks
            # Can happen for refinement level differences where it takes multiple iterations
            # for higher refined blocks to reach the same height
            if cax == 2:
                # Create a slice to select all guard cells
                sl_loc = sl.copy()
                sl_loc[2 - cax] = slice(
                    sl_loc[2 - cax].start - 1,
                    sl_loc[2 - cax].stop - 1,
                    sl_loc[2 - cax].step
                )
                
                # Check that all guard cells are filled (!= 0.0)
                # Have to reduce array across all three dimensions as axis are kept from slicing
                # even if they are size of 1
                sel_blk_fullgc = np.all(pres_gc[cur_bid, *sl_loc] != 0.0, axis=(-3, -2, -1))
                # Convert current block list to array for slicing
                cur_bid = np.asarray(cur_bid)
                # Select blocks which have all guard cells and convert back to list
                cur_bid = cur_bid[sel_blk_fullgc].tolist()
            
            blocks_time.append(cur_bid)
            blocks_touch += cur_bid

            # Get pressure gradients to integrate
            pgrad_gc = [pgradx_gc[cur_bid], pgrady_gc[cur_bid], pgradz_gc[cur_bid]]

            # Integrate data in block along given axis
            pres_gc[cur_bid] = integrate_block(
                pres=pres_gc[cur_bid], pgrad=pgrad_gc,
                axis=cax, sl=sl, direction=cdir,
                debug=False
            )
            
            if cax == 2:
                blocks_not_updated = pres_gc[cur_bid, 1:9, 1:9, 1:9] <= 0.0
                if debug or True:
                    print('\tNr of cells not updated:', blocks_not_updated.sum())
                    if blocks_not_updated.sum() > 0:
                        print(
                            '\tBlocks not updated correctly:',
                            np.asarray(cur_bid)[np.any(blocks_not_updated, axis=(1, 2, 3))]
                        )
            
            # Update ghost cells of neighbours of current block
            pres_gc = fill_ghostcells(
                data_pad=pres_gc, gid=gid, bbox=bbox,
                sel=cur_bid, debug=False,
                parent_norm=par_norm
            )
            
            cur_bid = get_neighbour_bid(
                bid=cur_bid,
                gid=gid, axis=cax, direction=cdir,
                sl=sl
            )
            
            # Reset integration to include first value
            # GC should contain correct value now
            sl[2 - cax] = slice(1, 2, None)
            
            # If no more neighbours exist we must be at the edge of the domain
            if len(cur_bid) == 0:
                at_domain_edge = True
    
    print('\tBlocks touched:', np.unique(blocks_touch).size)
    
    return pres_gc #[:, 1:9, 1:9, 1:9]


# Integrate block
# GC value is required to exist
# axis uses value in xyz order (x=0, y=1, z=2)
# sl[2-axis] has to start at 1 (direction=1) or 8 (direction=-1)
def integrate_block(pres, pgrad, axis, sl, direction=1, debug=False):
    start = sl[2 - axis].start
    stop = sl[2 - axis].stop
    sl_next = sl.copy()
    
    sl_prev = sl.copy()
    sl_prev[2 - axis] = slice(start-direction, stop-direction, direction)
    
    nloops = 8 - (1 - start)
    if debug:
        print('Number of loops', nloops)
        print('Pressure shape:', pres.shape)
        print('Pgrad shape', pgrad[axis].shape)
    
    for i in range(nloops): # range(start, 8, direction):
        if debug:
            print('previous slice:\t', sl_prev)
            print('current slice:\t', sl_next)
            print('prev_pres', pres[:, *sl_prev])
            print('prev_step', pgrad[axis][:, *sl_prev])
            print('next_step', pgrad[axis][:, *sl_next])
        
        # Take value of previous cell
        pres[:, *sl_next] = pres[:, *sl_prev].copy()
        # Add half step of previous cell
        pres[:, *sl_next] += pgrad[axis][:, *sl_prev]
        # Add half step of current cell
        pres[:, *sl_next] += pgrad[axis][:, *sl_next]
        
        # Update previous slice with next slice
        sl_prev = sl_next.copy()
        
        # Get start and stop value of current slice
        start = sl_next[2 - axis].start
        stop = sl_next[2 - axis].stop
        
        # Update next slice
        sl_next[2 - axis] = slice(start + direction, stop + direction, direction)
    
    return pres


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


# Fill ghost cells of neighbour blocks
# This routine pushes the face of the block onto its neighbours
# Cases:
#   if neighbour exist:
#       parse data as it is to the neighbour
#       if neighbour has children
#           update children with data
#   if neighbour does not exist and parent exist
#       parse data to neighbour of parent
def fill_ghostcells(data_pad, gid, bbox, sel=None, debug=False, parent_norm=4.0):
    # Check if sub selection is requested
    if sel is None:
        sel = range(data_pad.shape[0])

    # Make sure the block selection is either list or range
    if type(sel) is not list and type(sel) is not range:
        sel = [sel]

    # Slicing template
    # Used multiple times
    sl = [
        slice(None, None, None),
        slice(None, None, None),
        slice(None, None, None)
    ]
    
    #data_pad = data_pad.copy()
    #data_pad[:, 1:9, 1:9, 1:9] = update_all_parent_blocks(data=data_pad[:, 1:9, 1:9, 1:9], gid=gid)
    
    # Loop over requested blocks
    for it in sel:
        # Get block information (neighbours (6), parents (1), children (8))
        blk_info = gid[it] - 1
        if debug:
            print('blk_info:')
            print('\tneigh:', blk_info[:6])
            print('\tparent:', blk_info[6])
            print('\tchildren:', blk_info[7:])

        # Select list index of neighbours
        neigh_list = blk_info[:6]
        
        # Loop over all neighbours
        # 0   1   2   3   4   5
        # -x  +x  -y  +y  -z  +z
        for ni in range(6):
            # If neighbour is out of bounds no need to update
            # Take Fortran indexing into account
            if neigh_list[ni] == -39 - 1:
                continue
            
            # Get list index of neighbour
            tmp_neigh = neigh_list[ni]
            
            # Select slice perpendicular to x, y or z-axis
            # Count backwards as data order is z,y,x
            # ni     = [0, 1, 2, 3, 4, 5]
            # xyz_sw = [2, 2, 1, 1, 0, 0]
            # 0 - x, 1 - y, 2 - z
            xyz_sw = ni // 2
            
            # Update neighbour slice
            # Remove already determined slice
            sl_ind = [2, 1, 0]
            # ni   = [0, 1, 2, 3, 4, 5]
            # ax_n = [2, 2, 1, 1, 0, 0]
            ax_n = sl_ind.pop(xyz_sw)
            
            # Get slice index of data to pushed
            # to neighbour
            #   slice 1 for negative neighbour
            #   slice 8 for positive neighbour
            # in padded coordinates (+1)
            pm_sw_curr = ni % 2 * 7 + 1
            
            # Create slicing set for data from current block
            sl_curr = sl.copy()
            sl_curr[ax_n] = slice(pm_sw_curr, pm_sw_curr+1, None)
            sl_curr[sl_ind[0]] = slice(1, 9, None)
            sl_curr[sl_ind[1]] = slice(1, 9, None)
            
            if debug:
                print('\tSlices', sl_ind)
                print('\txyz_sw', xyz_sw, '\tax_n', ax_n)
            
            # Determine slice of neighbour
            # where data is supposed to be put
            pm_sw_neigh = (ni-1) % 2 * 9
            
            sl_neigh = sl.copy()
            sl_neigh[ax_n] = slice(pm_sw_neigh, pm_sw_neigh+1, None)
            sl_neigh[sl_ind[0]] = slice(1, 9, None)
            sl_neigh[sl_ind[1]] = slice(1, 9, None)
            
            # Select data to parse to neighbour
            # Sub selected without ghost cells
            face_data = data_pad[it, *sl_curr]
            
            # If neighbour exist pass data to neighbour
            if tmp_neigh != -1 - 1:
                if debug:
                    print(it, 'has neighbour:', tmp_neigh)
                    print('\torig:', sl_curr)
                    print('\tdest:', sl_neigh)
                    # print(face_data)
                # Store data in ghost cells of neighbour block
                data_pad[tmp_neigh, *sl_neigh] = face_data
            
                # Get children ids
                blk_info_neigh = gid[tmp_neigh] - 1
                children_neigh = blk_info_neigh[7:]
                neigh_has_child = children_neigh[0] != -1 - 1
                
                # If neighbour has children
                # Pass data to children of neighbour
                if neigh_has_child:
                    # Make a copy of the data that needs to be pushed
                    tmp_data = face_data.copy()
                    
                    # Expand array by factor 2 for easier slicing onto the children
                    tmp_data = np.repeat(tmp_data, 2, axis=sl_ind[0])
                    tmp_data = np.repeat(tmp_data, 2, axis=sl_ind[1])
                    
                    # Select children on the negative or positive side
                    # depending on the index of the neighbour
                    # ni = [0, 1, 2, 3, 4, 6]
                    # pm_sw_cid = [1, 0, 1, 0, 1, 0]
                    # 0 - negative side of neighbour
                    # 1 - positive side of neighbour
                    pm_sw_cid = (ni-1) % 2
                    
                    # Loop over children
                    for cid in range(8):
                        # Select 4 children which are neighbours
                        # Children are in z-order (xyz):
                        #   (0 0 0), (1 0 0), (0 1 0), (1 1 0)
                        #   (0 0 1), (1 0 1), (0 1 1), (1 1 1)
                        # Selection depends on axis along we check for a neighbour:
                        #   ax_n: 2 = x, 1 = y, 0 = z
                        #   cid     :  0  1  2  3  4  5  6  7
                        #   ax_n = 0: [0, 0, 0, 0, 1, 1, 1, 1]
                        #   ax_n = 1: [0, 0, 1, 1, 0, 0, 1, 1]
                        #   ax_n = 2: [0, 1, 0, 1, 0, 1, 0, 1]
                        #   0 - child is at negative position
                        #   1 - child is at positive position
                        if cid // 2**(2-ax_n) % 2 == pm_sw_cid:
                            # Get list index of child which lies
                            # at the correct side along the required axis
                            tmp_cid = children_neigh[cid]
                            # Determine shift (which of the 4 panels we get from the child)
                            shift_xyz = [
                                (cid // 4) % 2,
                                (cid // 2) % 2,
                                cid % 2,
                            ]
                            
                            shift0 = shift_xyz[sl_ind[0]] * 8
                            shift1 = shift_xyz[sl_ind[1]] * 8
    
                            # Reduce slices
                            # We only determine one quadrant now
                            sl_curr_cid = sl.copy()
                            sl_curr_cid[sl_ind[0]] = slice(0 + shift0, 8 + shift0, None)
                            sl_curr_cid[sl_ind[1]] = slice(0 + shift1, 8 + shift1, None)
                            
                            # Store data in child of neighbour block
                            data_pad[tmp_cid, *sl_neigh] = tmp_data[*sl_curr_cid]
                            if debug:
                                print('\tNeighbour', tmp_neigh, 'has child:', tmp_cid)
                                print('\t', tmp_cid, 'is child nr', cid, 'of', tmp_neigh)
                                print('\t\torig:', sl_curr_cid)
                                print('\t\tdest:', sl_neigh)
                                print('\n')
            
            # If no neighbour exists we have to pass data
            # to neighbour of parent in a down sampled version
            # Only possible if a parent exists
            if blk_info[6] != -1 - 1 and tmp_neigh == -1 - 1:
                if debug:
                    print('Parent exist:', blk_info[6])
                
                # Get parent block information
                blk_info_par = gid[blk_info[6]] - 1
                # Get neighbour block of parent (same direction as before)
                tmp_neigh_par = blk_info_par[:6][ni]
                
                # Check if neighbour is out of domain
                if tmp_neigh_par != -39 - 1:
                    if debug:
                        print('\tHas neighbour:', tmp_neigh_par)
                    
                    # Get child index in regard to parent block (relative 0-7)
                    cid = np.arange(8)[blk_info_par[7:] == it][0]
                    
                    # Determine the quadrant in which we parse the data
                    #sl_ind_cid = [2, 1, 0]
                    #sl_ind_cid.pop(xyz_sw)
                    
                    #shift_all = (bbox[tmp_neigh_par, :, 0] - bbox[it, :, 0])
                    #shift_all /= (bbox[it, :, 1] - bbox[it, :, 0])
                    #shift_all = np.abs(shift_all).astype(int) * 4
                    
                    # Determine shift (which of the 4 panels we get from the child)
                    #shift0 = (bbox[tmp_neigh_par, sl_ind[0], 0] - bbox[it, sl_ind[0], 0])
                    #shift0 /= (bbox[it, sl_ind[0], 1] - bbox[it, sl_ind[0], 0])
                    #shift0 = abs(int(np.round(shift0))) // 2 * 4
                    
                    #shift1 = (bbox[tmp_neigh_par, sl_ind[1], 0] - bbox[it, sl_ind[1], 0])
                    #shift1 /= (bbox[it, sl_ind[1], 1] - bbox[it, sl_ind[1], 0])
                    #shift1 = abs(int(np.round(shift1))) // 2 * 4
                    
                    #shift_sc = [1, 2]
                    
                    #shift0 = cid // (2 ** shift_sc[0]) % 2 * 4
                    #shift1 = cid // (2 ** shift_sc[1]) % 2 * 4
                    
                    shift_xyz = [
                        (cid // 4) % 2,
                        (cid // 2) % 2,
                        cid % 2,
                    ]
                    
                    shift0 = shift_xyz[sl_ind[0]] * 4
                    shift1 = shift_xyz[sl_ind[1]] * 4
                    
                    if shift0 != 0 and shift0 != 4:
                        debug = True
                    if shift1 != 0 and shift1 != 4:
                        debug = True
                    
                    if debug:
                        print('\tID of child:', cid)
                        print('\tRemaining slices:', sl_ind)
                        print('\tShape of data:', face_data.shape)
                        print('\tNeighbour:', tmp_neigh_par, '\tCurrent block', it)
                        print('\tshift0:', shift0, '\tshift1:', shift1)
                        #print('\tshift_all:', shift_all)
                    
                    # Reduce slices
                    # We only determine one quadrant now
                    sl_neigh[sl_ind[0]] = slice(1+shift0, 5+shift0, None)
                    sl_neigh[sl_ind[1]] = slice(1+shift1, 5+shift1, None)
                    
                    tmp_sh = list(face_data.shape)
                    
                    # Get new shape of subset
                    tmp_sh[sl_ind[0]] //= 2
                    tmp_sh[sl_ind[1]] //= 2
                    
                    # Downsample data
                    # Currently wrong, only considers 4 instead of 8 cells to fill parent
                    #sl_curr2 = sl.copy()
                    #if pm_sw_curr == 1:
                    #    sl_curr2[ax_n] = slice(pm_sw_curr, pm_sw_curr + 2, None)
                    #else:
                    #    sl_curr2[ax_n] = slice(pm_sw_curr - 1, pm_sw_curr + 1, None)
                        
                    #sl_curr2[sl_ind[0]] = slice(1, 9, None)
                    #sl_curr2[sl_ind[1]] = slice(1, 9, None)
                    #face_data = data_pad[it, *sl_curr2].mean(axis=ax_n, keepdims=True)

                    #print(sl_neigh, sl_curr2)

                    tmp_data = np.zeros(tmp_sh)
                    sl_ds = sl.copy()
                    
                    sl_ds[sl_ind[0]] = slice(0, None, 2)
                    sl_ds[sl_ind[1]] = slice(0, None, 2)
                    tmp_data += face_data.copy()[*sl_ds]
                    
                    sl_ds[sl_ind[0]] = slice(0, None, 2)
                    sl_ds[sl_ind[1]] = slice(1, None, 2)
                    tmp_data += face_data.copy()[*sl_ds]
                    
                    sl_ds[sl_ind[0]] = slice(1, None, 2)
                    sl_ds[sl_ind[1]] = slice(0, None, 2)
                    tmp_data += face_data.copy()[*sl_ds]
                    
                    sl_ds[sl_ind[0]] = slice(1, None, 2)
                    sl_ds[sl_ind[1]] = slice(1, None, 2)
                    tmp_data += face_data.copy()[*sl_ds]
                    
                    tmp_data /= parent_norm
                    
                    if debug:
                        print('\tpass data to:', tmp_neigh_par)
                        print('\torig:', sl_curr)
                        print('\tdest', sl_neigh)
                        print('\n')
                        # print(face_data)
                    
                    # Store data in ghost cells of neighbour block
                    data_pad[tmp_neigh_par, *sl_neigh] = tmp_data

    return data_pad
    
# sel_cid: Select children
# only corner, edge or all children at interface
# (0, 1, 2, 3) in z-order
# axis          0               1               2
# cids (dir=-1) (0, 2, 4, 6)    (0, 1, 4, 5)    (0, 1, 2, 3)
# cids (dir=+1) (1, 3, 5, 7)    (2, 3, 6, 7)    (4, 5, 6, 7)
# corner
def get_neighbour_bid(bid, gid, axis, direction, sl=None, debug=False):
    # Input is three slices
    # Select the correct children which could be hit
    child_sub_sel = [0, 1, 2, 3]
    if type(sl) is list:
        if len(sl) == 3:
            # Create copy of slices
            tmp_sl = sl.copy()
            # Remove axis along we look for neighbours
            tmp_sl.pop(2 - axis)
            child_sub_sel = []
            if tmp_sl[1].start < 5:
                if tmp_sl[0].start < 5:
                    child_sub_sel += [0]
                if tmp_sl[0].stop > 4:
                    child_sub_sel += [2]
            if tmp_sl[1].stop > 4:
                if tmp_sl[0].start < 5:
                    child_sub_sel += [1]
                if tmp_sl[0].stop > 4:
                    child_sub_sel += [3]
    
    if debug:
        print('\tWhich children in sub selection will be returned:', child_sub_sel)
    
    # Get correct entry in list
    neigh_ind = 2 * axis
    if direction == 1:
        neigh_ind += 1
    
    if type(bid) is np.ndarray:
        bid = bid.tolist()
    
    if type(bid) is not list:
        bid = [bid]
    
    
    # Get block info
    blk_info = gid[bid] - 1
    
    # Get list of neighbours
    # Remove 1 as Fortran starts index at 1
    bid_neigh = blk_info[:, neigh_ind]
    
    # If no neighbour exist at this level
    # we have to go one level up
    sel_no_neigh = bid_neigh == -1 - 1
    if debug:
        print('No neighbour - take parent:', sel_no_neigh.sum())
    # Check if any block doesn't have a neighbour
    if sel_no_neigh.sum():
        # Get list index of the parent
        blk_par = blk_info[sel_no_neigh][:, 6]
        # Get block info of parents
        blk_info_par = gid[blk_par] - 1
        # Get list index of neighbour block from parent
        bid_neigh_par = blk_info_par[:, neigh_ind]
        if debug:
            print('\tNew neighbours:', bid_neigh_par.size)
            print('\tUnique:', np.unique(bid_neigh_par).size)
        # Update list with new neighbour
        bid_neigh[sel_no_neigh] = bid_neigh_par
    
    # Check if we reached domain edge
    sel_at_boundary = bid_neigh == -39 - 1
    # Remove these blocks from the list
    if debug:
        print('No neighbour - at the domain edge:', sel_at_boundary.sum())
    if sel_at_boundary.sum():
        bid_neigh = bid_neigh[np.logical_not(sel_at_boundary)]
    
    # If neighbour has children replace these bids
    # with those of the children
    sel_has_child = gid[bid_neigh][:, -1] != -1
    if debug:
        print('Has children:', sel_has_child.sum())
    if sel_has_child.sum():
        # Select the children indices which are at the interface with the input block
        sel_correct_child = np.arange(8) // 2**axis % 2 == (-direction + 1) // 2
        if debug:
            print('Select', sel_correct_child.sum(), 'new children')
        # Get children of blocks which have children
        child_bids = gid[bid_neigh][sel_has_child][:, 7:][:, sel_correct_child] - 1
        # Create block list of neighbours without children
        bid_neigh = bid_neigh[np.logical_not(sel_has_child)].tolist()
        # Add children of the neighbours which have children
        bid_neigh += child_bids[:, child_sub_sel].flatten().tolist()
    
    # If not a list, convert array to list
    if type(bid_neigh) is not list:
        bid_neigh = bid_neigh.tolist()
    
    if debug:
        print('\tTotal blocks found:', len(bid_neigh))
    return np.unique(bid_neigh).tolist()

