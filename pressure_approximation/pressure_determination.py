import numpy as np
from itertools import permutations
from io_amr import read_settings
from constants import *
from integration import integ_acc_order, integ_acc_order_diag

def get_pressures(acx, acy, acz, dens, dx, dy, dz, all_corners=True):
    print('Get pressures using ordered integration steps')
    rev_set = set(list(permutations([True, False] * 3, 3)))
    if all_corners != True:
        rev_set = [(False, False, False)]
    
    pres_xyz = np.zeros_like(dens)
    # pres_xzy = np.zeros_like(dens)
    pres_xzy = 0.0
    pres_yxz = np.zeros_like(dens)
    # pres_yzx = np.zeros_like(dens)
    pres_yzx = 0.0
    # pres_zxy = np.zeros_like(dens)
    pres_zxy = 0.0
    # pres_zyx = np.zeros_like(dens)
    pres_zyx = 0.0
    
    low = [0, 0, 0]
    high = [-1, -1, -1]
    
    for s in rev_set:
        cor = np.asarray(low)
        cor[list(s)] += np.asarray(high)[list(s)]
        print('\tCorner:', cor)
        # Determine pressure for all 6 integration orders
        # XYZ
        pres_xyz += integ_acc_order(
            accx=acx, accy=acy, accz=acz, rho=dens,
            dx=dx, dy=dy, dz=dz, order=[0, 1, 2],
            reverse=list(s)
        )
        # XZY
        # pres_xzy += integ_acc_order(
        #    accx=acx, accy=acy, accz=acz, rho=dens,
        #    dx=dx, dy=dy, dz=dz, order=[0, 2, 1],
        #    reverse=list(s)
        # )
        # YXZ
        pres_yxz += integ_acc_order(
            accx=acx, accy=acy, accz=acz, rho=dens,
            dx=dx, dy=dy, dz=dz, order=[1, 0, 2],
            reverse=list(s)
        )
        # YZX
        # pres_yzx += integ_acc_order(
        #    accx=acx, accy=acy, accz=acz, rho=dens,
        #    dx=dx, dy=dy, dz=dz, order=[1, 2, 0],
        #    reverse=list(s)
        # )
        # ZXY
        # pres_zxy += integ_acc_order(
        #    accx=acx, accy=acy, accz=acz, rho=dens,
        #    dx=dx, dy=dy, dz=dz, order=[2, 0, 1],
        #    reverse=list(s)
        # )
        # ZYX
        # pres_zyx += integ_acc_order(
        #    accx=acx, accy=acy, accz=acz, rho=dens,
        #    dx=dx, dy=dy, dz=dz, order=[2, 1, 0],
        #    reverse=list(s)
        # )
    
    pres_xyz /= float(len(rev_set))
    # pres_xzy /= float(len(rev_set))
    pres_yxz /= float(len(rev_set))
    # pres_yzx /= float(len(rev_set))
    # pres_zxy /= float(len(rev_set))
    # pres_zyx /= float(len(rev_set))
    
    print('PRES_XYZ (min, max):', pres_xyz.min(), pres_xyz.max())
    # print('PRES_XZY (min, max):', pres_xzy.min(), pres_xzy.max())
    print('PRES_YXZ (min, max):', pres_yxz.min(), pres_yxz.max())
    # print('PRES_YZX (min, max):', pres_yzx.min(), pres_yzx.max())
    # print('PRES_ZXY (min, max):', pres_zxy.min(), pres_zxy.max())
    # print('PRES_ZYX (min, max):', pres_zyx.min(), pres_zyx.max())
    
    # Mean between orders which have z at the same position
    # in integration order
    pres_rrz = (pres_xyz + pres_yxz) / 2.
    pres_rzr = (pres_xzy + pres_yzx) / 2.
    pres_zrr = (pres_zxy + pres_zyx) / 2.
    
    print()
    print('PRES_RRZ (min, max):', pres_rrz.min(), pres_rrz.max())
    # print('PRES_RZR (min, max):', pres_rzr.min(), pres_rzr.max())
    # print('PRES_ZRR (min, max):', pres_zrr.min(), pres_zrr.max())
    
    return pres_zrr, pres_rzr, pres_rrz


def get_pressures_diag(acx, acy, acz, dens, dx, dy, dz, all_corners=True):
    rev_set = set(list(permutations([True, False] * 3, 3)))
    if all_corners != True:
        rev_set = [(False, False, False)]
    
    pres_diag = np.zeros_like(dens)
    for s in rev_set:
        # Determine pressure for all 6 integration orders
        # XYZ
        pres_diag += integ_acc_order_diag(
            accx=acx, accy=acy, accz=acz, rho=dens,
            dx=dx, dy=dy, dz=dz,
            reverse=list(s)
        )
    
    pres_diag /= float(len(rev_set))
    print('PRES_DIAG (min, max):', pres_diag.min(), pres_diag.max())
    
    return pres_diag


def fit_press(prof, x, y, z, fname, order, ax=None):
    pc2cm = const.parsec / const.centi
    # Get cell coordinates for all cells
    xx, yy, zz = np.meshgrid(x / pc2cm, y / pc2cm, z / pc2cm, indexing='ij')
    
    # Determine cylindrical radius
    rr = (xx ** 2 + yy ** 2) ** 0.5
    
    sim_set = read_settings('settings.txt')
    # nx = sim_set['nx']
    # ny = sim_set['ny']
    # nz = sim_set['nz']
    nz = z.size
    
    # Lower density limit [g cm^-3]
    low_limit = sim_set['low_limit']
    # Total gas mass [g]
    gas_mass = sim_set['gas_mass'] * m_solar
    print('Gas mass: ', gas_mass)
    
    # Scaling factors for density profile [cm]
    a_sc = sim_set['a']  # * const.parsec / const.centi
    b_sc = sim_set['b']  # * const.parsec / const.centi
    print('Scaling factor a: ', a_sc)
    print('Scaling factor b: ', b_sc)
    
    # Central density of profile
    rho0 = gas_mass / (2 * np.pi * (a_sc * pc2cm) ** 2 * b_sc * pc2cm) * 0.5 ** 2
    
    # Determine ratio between central and threshold density
    rho_ratio = rho0 / low_limit
    print('rat', rho_ratio)
    # Scaled for z
    rho_ratio_z = rho_ratio / np.cosh(zz / b_sc)
    # Scaled for r
    rho_ratio_r = rho_ratio / np.cosh(rr / a_sc)
    
    # Limiting radius where we reach density threshold
    r_lim = a_sc * np.log(rho_ratio_z + np.sqrt(rho_ratio_z ** 2 - 1.))
    # Limiting height where we reach density threshold
    z_lim = b_sc * np.log(rho_ratio_r + np.sqrt(rho_ratio_r ** 2 - 1.))
    
    # Filter nan and set to 0
    # Density threshold should never be at center
    r_lim = np.nan_to_num(r_lim)
    z_lim = np.nan_to_num(z_lim)
    
    print(r_lim.min(), r_lim.max())
    # Index of midplane
    z_mid = nz // 2
    # Shift needed for odd and even nz
    if nz % 2 == 0:
        z_shift = 1
    else:
        z_shift = 0
    
    # Loop over all heights
    # + 1 ensures that we catch all heights
    # even for odd number of cells
    # 3 // 2 = 1, (3+1)//2 = 2
    
    fits_in = []
    fits_out = []
    for i in range((nz + 1) // 2):
        if i == 0 and nz % 2 == 1:
            # Only midplane for odd number of z layers
            sl = (slice(None), slice(None), z_mid)
        else:
            # Else use both layers, above and below midplane
            # Shift upper layer down for even nz
            sl = (slice(None), slice(None), [z_mid - (i + z_shift), z_mid + i])
        
        # All cells which lie within limiting radius
        # for current slice
        tr = rr[sl] <= r_lim[sl]
        if tr.sum() > 0:
            fit_z_in = np.polyfit(
                (rr[sl][tr] ** 2 + zz[sl][tr] ** 2) ** 0.5,
                prof[sl][tr],
                deg=order
            )
            if ax is not None:
                p_temp = np.poly1d(fit_z_in)
                r_fit = np.linspace(0, r_lim[sl].max(), 100)
                r_fit = (r_fit ** 2 + zz[sl].max() ** 2) ** 0.5
                ax.plot(r_fit, p_temp(r_fit))
        else:
            fit_z_in = np.zeros(order + 1)
        
        # All cells which lie outside limiting radius
        # for current slice
        tr = rr[sl] >= r_lim[sl]
        if tr.sum() > 0:
            fit_z_out = np.polyfit(
                (rr[sl][tr] ** 2 + zz[sl][tr] ** 2) ** 0.5,
                prof[sl][tr],
                deg=order
            )
            if ax is not None:
                p_temp = np.poly1d(fit_z_out)
                r_fit = np.linspace(r_lim[sl].max(), rr[sl].max(), 100)
                r_fit = (r_fit ** 2 + zz[sl].max() ** 2) ** 0.5
                ax.plot(r_fit, p_temp(r_fit))
        else:
            fit_z_out = np.zeros(order + 1)
        
        fits_in.append(fit_z_in)
        fits_out.append(fit_z_out)
    
    if ax is not None:
        ax.scatter(
            (rr ** 2 + zz ** 2) ** 0.5,
            prof,
            s=0.2, c=zz, marker='x'
        )
        ax.set_yscale('log')
        return
        
        for j in range(z.size // 2 - 1):
            for i in range(9):
                ilow = j
                
                ins = np.poly1d(fits_in[ilow])
                outs = np.poly1d(fits_in[ilow + 1])
                ins2 = np.poly1d(fits_out[ilow])
                outs2 = np.poly1d(fits_out[ilow + 1])
                
                rrr = np.linspace(0, 4000, 1000)
                
                ilow = z.size // 2 + ilow
                ##r_th_low = r_lim[z.size//2, z.size//2, ilow]
                # r_th_high = r_lim[z.size//2, z.size//2, ilow+1]
                # r_th_low = (r_th_low + r_th_high) / 2.
                # rrr = np.linspace(0, r_th_low, 100)
                # rrr2 = np.linspace(r_th_low, 4000, 100)
                zmid = ((i + 1) * z[ilow] + (9 - i) * z[ilow + 1]) / 10. / pc2cm
                
                r_th_low = rho_ratio / np.cosh(z[ilow] / pc2cm / b_sc)
                r_th_low = a_sc * np.log(r_th_low + np.sqrt(r_th_low ** 2 - 1.))
                
                r_th_high = rho_ratio / np.cosh(z[ilow + 1] / pc2cm / b_sc)
                r_th_high = a_sc * np.log(r_th_high + np.sqrt(r_th_high ** 2 - 1.))
                
                r_th = rho_ratio / np.cosh(zmid / b_sc)
                r_th = a_sc * np.log(r_th + np.sqrt(r_th ** 2 - 1.))
                
                lowin = ins((rrr ** 2 + (z[ilow] / pc2cm) ** 2) ** 0.5)
                lowout = ins2((rrr ** 2 + (z[ilow] / pc2cm) ** 2) ** 0.5)
                highin = outs((rrr ** 2 + (z[ilow + 1] / pc2cm) ** 2) ** 0.5)
                highout = outs2((rrr ** 2 + (z[ilow + 1] / pc2cm) ** 2) ** 0.5)
                
                # lowin = ins((rrr**2 + (zmid)**2)**0.5)
                # lowout = ins2((rrr**2 + (zmid)**2)**0.5)
                # highin = outs((rrr**2 + (zmid)**2)**0.5)
                # highout = outs2((rrr**2 + (zmid)**2)**0.5)
                
                rsph = ((rrr ** 2 + zmid ** 2) ** 0.5 - (rrr ** 2 + (z[ilow] / pc2cm) ** 2) ** 0.5)
                ddz = ((rrr ** 2 + (z[ilow + 1] / pc2cm) ** 2) ** 0.5 - (rrr ** 2 + (z[ilow] / pc2cm) ** 2) ** 0.5)
                rsph /= ddz
                
                # mean = np.where(rrr < r_th, lowin, lowout) + (np.where(rrr < r_th, highin, highout) - np.where(rrr < r_th, lowin, lowout)) * rsph
                # mean = np.where((rrr > r_th_high) * (mean > lowout+(highout-lowout) *rsph), lowout+(highout-lowout) *rsph, mean)
                
                # tr = lowin > 0 and lowin < lowout
                # tr2 = highin > 0 and highin < highout
                # mean = (
                #    (i+1) * np.log10(np.where(rrr < r_th, lowin, lowout))
                #    + (9-i) * np.log10(np.where(rrr < r_th, highin, highout))
                # ) / 10.
                
                mean = (
                        (1 - rsph) * np.log10(np.where(rrr < r_th, lowin, lowout))
                        + (rsph) * np.log10(np.where(rrr < r_th, highin, highout))
                )
                mean = 10 ** mean
                mean = np.nan_to_num(mean)
                
                # mean = (
                #    (1-rsph) * np.where(rrr < r_th, lowin, lowout)
                #    + (rsph) * np.where(rrr < r_th, highin, highout)
                # )
                
                # mean = np.where(
                #    (rrr < r_th_low) * (rrr > r_th_high) * (mean > ((i+1) * lowout + (9-i) * highout) / 10.),
                #    ((i+1) * np.log10(lowout) + (9-i) * np.log10(highout)) / 10.,
                #    np.log10(mean)
                # )
                
                z_th = rho_ratio / np.cosh(0 / a_sc)
                z_th = b_sc * np.log(z_th + np.sqrt(z_th ** 2 - 1.))
                
                mean = np.where(
                    (rrr < r_th_low) * (rrr > r_th_high) * (
                                mean > 10 ** ((1 - rsph) * np.log10(lowout) + (rsph) * np.log10(highout))) + (
                                z_th - zmid < ddz) + (mean == 0),
                    ((1 - rsph) * np.log10(lowout) + (rsph) * np.log10(highout)),
                    np.log10(mean)
                )
                mean = 10 ** mean
                
                # mean = np.where(
                #    (rrr < r_th_low) * (rrr > r_th_high) * (mean > ((1-rsph) * lowout + (rsph) * highout)) + (z_th - zmid < ddz),
                #    ((1-rsph) * lowout + (rsph) * highout),
                #    mean
                # )
                
                if np.isnan(mean).any() or (mean == 0.).any():
                    nantr = np.isnan(mean) + (mean == 0) + (mean < 0)
                    print(rsph[nantr], zmid, lowin[nantr], lowout[nantr], highin[nantr], highout[nantr])
                    print(mean[nantr])
                    # return
                # mean = np.where((rrr < r_th_low) * (rrr > r_th_high), , mean)
                
                # mean = (
                #    np.where(
                #            (lowout != 0) * (rrr <= r_th_low),
                #            np.minimum(lowin, lowout),
                #            lowin
                #    ) + np.where(
                #        (highout != 0) * (rrr <= r_th_low),
                #        np.minimum(highin, highout),
                #        highin
                #    )
                # ) / 2.
                
                # mean = (np.where(lowin > 0, lowin, lowout) + np.where(highin > 0, highin, highout)) / 2.
                # mean2 = (lowin[lowin > 0] + lout[lowin == 0] + highin[highin > 0] + highout[highin == 0]) / 2.
                # mean2 = (ins2((rrr2**2+(z[ilow]/pc2cm)**2)**0.5) + outs2((rrr2**2 + (z[ilow+1]/pc2cm)**2)**0.5)) / 2.
                ax.plot((rrr ** 2 + zmid ** 2) ** 0.5, mean, c='k')
            # ax.plot((rrr2**2 + zmid**2)**0.5, mean2, c='k', ls=':')
    
    print('Saving coefficients')
    np.savetxt(
        fname='%s.dat' % fname, delimiter=' ', fmt='%1.15e',
        X=fits_in, comments='',
        header='%i\n%i\n%s' % (
            order, (nz + 1) // 2,
            ' '.join(['%1.8e' % (it / 1e3) for it in zz.mean(axis=(0, 1))][z_mid:])
        )
    )
    
    np.savetxt(
        fname='const_%s.dat' % fname, delimiter=' ', fmt='%1.15e',
        X=fits_out, comments='',
        header='%i\n%i' % (
            order, (nz + 1) // 2
        )
    )
