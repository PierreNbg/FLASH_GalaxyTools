import numpy as np

# Integrates the acceleration profiles coupled with density
# along a specific order of axis.
# E.g:
# order=[0, 1, 2] means that we first integrate along x,
# then y and at last z while
# order=[1, 2, 0] mean that we first integrate along y,
# then z and at last x.
# Always starts at lower left corner [0, 0, 0]
def integ_acc_order(accx, accy, accz, rho, dx, dy, dz, order=None, reverse=[False, False, False]):
    # Compact acceleration for easier access with index
    # Order of
    acc = [accx, accy, accz]
    # Compact cell size for easier access with index
    dd = [dx, dy, dz]

    # Shape of data set
    sh = rho.shape
    # Output array with pressure
    pres = np.zeros_like(rho)

    if order is None:
        print('No order given, default to [0, 1, 2]')
        order = [0, 1, 2]

    # By default all arrays stay unmodified
    sls = [slice(None), slice(None), slice(None)]
    # Check through the orders if any of them should be flipped
    for i in range(len(order)):
        # If reversed order update slice of current axis
        # And negate dx/dy/dz to integrate backwards
        if reverse[i]:
            sls[i] = slice(None, None, -1)
            dd[i] = -dd[i]

    sls = tuple(sls)
    accx = accx[sls]
    accy = accy[sls]
    accz = accz[sls]
    rho = rho[sls]
    acc = [accx, accy, accz]

    # Loop over all entries in order list
    for j in range(len(order)):
        # Default index is lower left corner or (0, 0, 0)
        tr = [0, 0, 0]
        ind_start = 0
        if j == 1:
            # After first pass take edge along
            # which was already integrated
            tr[order[0]] = slice(None)

            # Skip first cell since already calculated
            ind_start = 1
        elif j == 2:
            # After second pass take face along
            # which was already integrated
            tr[order[0]] = slice(None)
            tr[order[1]] = slice(None)

            # Skip first cell since already calculated
            ind_start = 1

        # Loop over all cell along given axis
        for i in range(ind_start, sh[order[j]]):
            # Update current cell index / indices
            tr[order[j]] = i
            # Tuple for next cell(s) to compute
            tr_n = tuple(tr)
            # Get previous cell index / indices
            tr[order[j]] -= 1
            # Tuple of previous cell(s)
            tr_p = tuple(tr)

            # First integrate to cell center from boundary
            # Enforces a lower pressure value
            if j == 0 and i == 0:
                pres[tr_n] = acc[order[0]][tr_n] * rho[tr_n] * dd[order[0]] / 2.
                pres[tr_n] += acc[order[1]][tr_n] * rho[tr_n] * dd[order[1]] / 2.
                pres[tr_n] += acc[order[2]][tr_n] * rho[tr_n] * dd[order[2]] / 2.
            else:
                # Current pressure is previous pressure
                pres[tr_n] = pres[tr_p]
                # plus the half integral in the previous and current cell
                pres[tr_n] += acc[order[j]][tr_p] * rho[tr_p] * dd[order[j]] / 2.
                pres[tr_n] += acc[order[j]][tr_n] * rho[tr_n] * dd[order[j]] / 2.

    return pres[sls]

# Integrates the acceleration profiles coupled with density
# Diagonal approach
# NOT WORKING
def integ_acc_order_diag(accx, accy, accz, rho, dx, dy, dz, order=None, reverse=[False, False, False]):
    # Compact acceleration for easier access with index
    # Order of
    acc = [accx, accy, accz]
    # Compact cell size for easier access with index
    dd = [dx, dy, dz]

    # Shape of data set
    sh = rho.shape
    # Output array with pressure
    pres = np.zeros_like(rho, dtype=np.float128)

    # Get dimensions of array
    sx, sy, sz = rho.shape

    order = [0, 1, 2]
    # By default all arrays stay unmodified
    sls = [slice(None), slice(None), slice(None)]
    # Check through the orders if any of them should be flipped
    for i in range(len(order)):
        # If reversed order update slice of current axis
        # And negate dx/dy/dz to integrate backwards
        if reverse[i]:
            sls[order[i]] = slice(None, None, -1)
            dd[order[i]] = -dd[order[i]]

    print(sls, dd)
    sls = tuple(sls)
    accx = accx[sls]
    accy = accy[sls]
    accz = accz[sls]
    rho = rho[sls]
    acc = [accx, accy, accz]


    # Array containing the number of integrations in cell (normalisation value)
    # nr_int = np.zeros_like(rho, dtype=np.float128)
    # nr_arr = np.zeros_like(rho, dtype=int)

    # Go through all diagonal slices (3D)
    for sid in range(sx-1 + sy-1 + sz-1 + 1):
        # Determine all cell indicies at current slice index
        sid_list = []
        print('Current slice:', sid)
        # Loop over all possible i indicies
        # Reduce to minimum between current slice and max dimension in i-axis
        for i in range(min(sid+1, sx)):
            # Loop over all possible j indicies
            # Reduce to minimum between slice and max dimension in j-axis
            for j in range(min(sid-i+1, sy)):
                # k-index is remainder from slice index and i/j index
                # as the total sum should be conserved for a slice
                # Only use j-indicies which are within j-dimension
                if sid-i-j < sz and sid-i-j >= 0:
                    sid_list.append([i, j, sid-i-j])

        # Go across all cells in current slice
        for ijk in sid_list:
            # Get current index
            ii, jj, kk = ijk

            nr = 0
            dp = 0.
            # Only add x-contribution if previous cell along x-axis exists
            if ii != 0:
                nr += 1
                # Add the number of previous integration permutations to the current cell
                # nr_int[ii, jj, kk] += nr_int[ii-1, jj, kk]

                # Add pressure of previous cell
                pres[ii, jj, kk] += pres[ii-1, jj, kk]
                # Plus the half step of the previous cell
                pres[ii, jj, kk] += acc[0][ii-1, jj, kk] * rho[ii-1, jj, kk] * dd[0] / 2.
                # Add the half integral of the current cell to get to the cell center
                pres[ii, jj, kk] += acc[0][ii, jj, kk]   * rho[ii, jj, kk]   * dd[0] / 2.

            # Only add y-contribution if previous cell along y-axis exists
            if jj != 0:
                nr += 1
                # Add the number of previous integration permutations to the current cell
                # nr_int[ii, jj, kk] += nr_int[ii, jj-1, kk]

                # Add pressure of previous cell
                pres[ii, jj, kk] += pres[ii, jj-1, kk]
                # Plus the half step of the previous cell
                pres[ii, jj, kk] += acc[1][ii, jj-1, kk] * rho[ii, jj-1, kk] * dd[1] / 2.
                # Add the half integral of the current cell to get to the cell center
                pres[ii, jj, kk] += acc[1][ii, jj, kk]   * rho[ii, jj, kk]   * dd[1] / 2.

            # Only add z-contribution if previous cell along z-axis exists
            if kk != 0:
                nr += 1
                # Add the number of previous integration permutations to the current cell
                # nr_int[ii, jj, kk] += nr_int[ii, jj, kk-1]

                # Add pressure of previous cell
                pres[ii, jj, kk] += pres[ii, jj, kk-1]
                # Plus the half step of the previous cell
                pres[ii, jj, kk] += acc[2][ii, jj, kk-1] * rho[ii, jj, kk-1] * dd[2] / 2.
                # Add the half integral of the current cell to get to the cell center
                pres[ii, jj, kk] += acc[2][ii, jj, kk]   * rho[ii, jj, kk]   * dd[2] / 2.

            if ii==0 and jj==0 and kk==0:
                # Add the half integral of the current cell to get to the cell center
                pres[ii, jj, kk] += acc[0][ii, jj, kk] * rho[ii, jj, kk] * dd[0] / 2.
                pres[ii, jj, kk] += acc[1][ii, jj, kk] * rho[ii, jj, kk] * dd[1] / 2.
                pres[ii, jj, kk] += acc[2][ii, jj, kk] * rho[ii, jj, kk] * dd[2] / 2.

            if nr > 0:
                pres[ii, jj, kk] /= nr
                #nr_arr[ii, jj, kk] = nr

            # If there were no other contribtutions from adjacent cells set number of permutations to 1
            # if nr_int[ii, jj, kk] == 0:
            #     nr_int[ii, jj, kk] = 1

    return pres[sls] #, 1#, nr_arr
