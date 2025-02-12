def integrate_on_amr(acc, rho, bbox, bsize, rlvl, blist, bns, brefs):
    # Get sorting index to make sure everything is in z-order
    # Let's see if this is necessary
    sort_ind = np.argsort(blist)
    
    # Get the smallest cell size
    dx = (bbox[-1, :, 1] - bbox[0, :, 0]) / bns / 2 ** (brefs[0] - brefs[1] + 3)
    
    bbox = bbox[sort_ind]
    # Get block lower left corner index in uniform grid
    bind_cor = np.round((bbox[:, :, 0] - bbox[0, :, 0]) / dx).astype(int)
    
    bs_cells = np.round(bsize / dx).astype(int)
    
    # Unpack accelerations
    acx, acy, acz = acc
    # Sort accelerations into z-order
    acx = acx[sort_ind]
    acy = acy[sort_ind]
    acz = acz[sort_ind]
    
    #
    rho = rho[sort_ind]
    
    #
    #
    bsize = bsize[sort_ind]
    #
    rlvl = rlvl[sort_ind]


def get_block_from_pos(bind_cor, bs_cells, pos):
    cond = (pos >= bind_cor).prod(axis=1)
    cond *= (pos < bind_cor + bs_cells).prod(axis=1)
    blk_id = np.where(cond.astype(bool))
    return blk_id


def get_value_at_pos(bind_cor, bs_cells, pos, data):
    blk_id = get_block_from_pos(bind_cor=bind_cor, bs_cells=bs_cells, pos=pos)
    cell_id = (pos - bind_cor[blk_id]) // bs_cells[blk_id]
    data_cell = data[blk_id, *cell_id]
    return data_cell


def get_neighbour_face(blk_id, bind_cor, bs_cells, data, axis, direction):
    neigh_pos = bind_cor[blk_id].copy()
    if direction == -1:
        neigh_pos[axis] -= 1
    else:
        neigh_pos[axis] += bs_cells[blk_id]
    # get_neighbour(bind_cor=bind_cor, curr)


def get_neighbour(bind_cor, cur_loc, bs_cells, axis, direction=1):
    # Make axis selection
    ax_sel = [0, 1, 2]
    cax = ax_sel.pop(axis)
    
    # Get current block position
    # current_loc = bind_cor[cblock_id]
    # Next block positions is at current block position
    next_loc = cur_loc.copy()
    # plus offset in number of cells (in uniform grid)
    # next_loc[cax] += bs_cells[cblock_id][cax] * direction
    next_loc[cax] += direction
    if np.any(next_loc < 0.0):
        raise ValueError(
            'Next location is out of bounds in axis=%i. '
            'Next location would have been: [%i, %i, %i].' %
            (cax, *next_loc)
        )
    # Find next block.
    # Next position has to lie in one block...,
    cond = (next_loc >= bind_cor).prod(axis=1)
    cond *= (next_loc < bind_cor + bs_cells).prod(axis=1)
    nblock_id = np.where(cond.astype(bool))
    
    print(next_loc, nblock_id)
    try:
        return nblock_id[0][0]
    except IndexError:
        raise IndexError(
            'Next location is out of bounds in axis=%i. '
            'Next location would have been: [%i, %i, %i].' %
            (cax, *next_loc)
        )


def get_neighbour_id(bid, gid, axis, pm=1):
    # Get list index of neighbours of current block
    block_info = gid[bid, :6]
    # Select requested neighbour (-1 to account for Fortran indexing)
    neigh_id = block_info[axis * 2 + (pm + 1) // 2] - 1
    return neigh_id
