import glob
import os
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

def prepare_fiber(fiber):
    fibermean = fiber.mean(axis=0)
    fiberstd = fiber.std(axis=0)
    new = (fiber-fibermean)/fiberstd
    return new, fibermean, fiberstd

def finish(fiber, fibermean, fiberstd):
    new = (fiber+fibermean)*fiberstd
    return new

def get_scprod(fiber, ncomp):
    fiber[np.isnan(fiber)] = 0
    covmat = np.cov(fiber.T)
    eigenvals, eigenvecs = np.linalg.eig(covmat)
    eigenvals, eigenvecs = np.real(eigenvals), np.real(eigenvecs)
    eigenpairs = [(np.abs(eigenvals[i]), eigenvecs[:,i]) for i in np.argsort(abs(eigenvals))][::-1]
    imp = eigenpairs[:ncomp]
    imp = [x[1] for x in imp]
    imp = np.array(imp)
    scprod = imp.dot(fiber.T)
    scprod = np.array([scprod.T[i]/np.linalg.norm(fiber[i])**2 for i in range(scprod.T.shape[0])]).T
    fiber_pca = fiber.dot(imp.T)
    return scprod, fiber_pca

def get_quasi_fiber(fiber, scprod, fiber_pca, fibermean, fiberstd):
    for x in fiber:
        x = x/(np.linalg.norm(x))
    quasi = scprod.dot(fiber)
    norm = np.linalg.norm
    for i in range(quasi.shape[0]):
        quasi[i] = quasi[i]/norm(quasi[i])
    new_fiber = (np.dot(fiber_pca, quasi)+fibermean)*fiberstd
    return new_fiber

def get_polyerror(fiber, fiber_new, index):
    f = fiber[index]
    fn = fiber_new[index]
    re = (f-fn)/f
    x = np.arange(0,re.shape[0],1)
    sigma = np.nanstd(re)
    kappa = 3
    flag = np.where(np.isfinite(re))
    pp = np.polyfit(x[flag], re[flag],5)
    poly = pp[5]+x*pp[4]+x*x*pp[3]+x*x*x*pp[2]+x*x*x*x*pp[1] + x*x*x*x*x*pp[0]
    repoly = re - poly
    return re, repoly

def get_fibers (shots, amp, compareamp): #(nights, sigma, amp, compareamp) # second version of get_fibers : gets lists of first and second fibers
    nights = np.unique([x[:-4] for x in shots])
    patterns = ['/work/03946/hetdex/maverick/red1/reductions/{}/virus/virus0000*/exp0?/virus/multi*.fits'.format(night) for night in nights]

    ff = np.array([])
    for pattern in patterns:
        ff = np.append(ff, np.array(glob.glob(pattern)))

    ifuslots = []
    amps = []
    exposures = []
    allshots = []
    nights = []

    for fff in ff:
        h,t = os.path.split(fff)
        ifuslots.append( t[10:13] )
        amps.append( t[18:20] )
        exposures.append( h.split('/')[-2])
        allshots.append(h.split('/')[-3])
        nights.append(h.split('/')[-5])

    ifuslots = np.array(ifuslots)
    amps = np.array(amps)
    exposures = np.array(exposures)
    allshots = np.array(allshots)
    nights = np.array(nights)
    ee, ss, aa, shs = np.unique(exposures), np.unique(ifuslots), np.unique(amps), np.unique(shots)# np.unique(['{}v{}'.format(night, x[-3:]) for night,x in zip(nights,allshots)])

    goodshotmask = np.ones(len(shs), dtype=np.int8)*True

        # only use shots with more than one exposure, i.e. hetdex shots
    for shot in shs:
        exp = ee[2] # second exposure: 'exp03'
        ii = (exposures == exp) * (allshots == 'virus0000{}'.format(shot[-3:]))
        if not sum(ii) >= 1: # i.e. if there is no second exposure
            print('{} has no exposure {}'.format(shot, exp))
            goodshotmask[np.where(shs==shot)] = False
            continue
        fin = ff[ii][0]
        print("Reading {}".format(fin))
        try:
            hdu = fits.open(fin)
            a = hdu['sky_spectrum'].data
            hdu.close()
        except KeyError:
            hdu.close()
            goodshotmask[np.where(shs==shot)] = False
            print('{} has no sky spectrum'.format(shot))
        shots = shs[np.where(goodshotmask)]
    shots = np.array(shots)
    print('These shots are going to be used: ', shots)

    allgoodshotsmask = np.ones(len(allshots), dtype=np.int8)*False
    allshots = np.array(['{}v{}'.format(night, x[-3:]) for night,x in zip(nights,allshots)])
    for shot in shots:
        allgoodshotsmask[np.where(allshots==shot)] = True

    #print('allgoodshotsmask: ', allgoodshotsmask)
        # these arrays are going to be used instead of dictionary keys
    ff, allshots, exposures, ifuslots, amps = ff[np.where(allgoodshotsmask)], allshots[np.where(allgoodshotsmask)],exposures[np.where(allgoodshotsmask)], ifuslots[np.where(allgoodshotsmask)], amps[np.where(allgoodshotsmask)]
    nights = nights[np.where(allgoodshotsmask)]
    ee, ss, aa, shs = np.unique(exposures), np.unique(ifuslots), np.unique(amps), np.unique(shots)
    reihenfolge = np.argsort(ff)
    ff, allshots, exposures, ifuslots, amps = ff[reihenfolge], allshots[reihenfolge],exposures[reihenfolge], ifuslots[reihenfolge], amps[reihenfolge]

    ifus = ss[:int(ss.shape[0]/2)]
    compareifus = ss[int(ss.shape[0]/2):]

    for night in np.unique(nights):
        for exp in ee:
            for ifu in ifus:
                ii = (nights==night)*(ifuslots==ifu)*(exposures==exp)
                if sum(ii) == 0:
                    ifus = ifus[np.where(ifus!=ifu)]
                    print('{} will not be used.'.format(ifu))
            for ifu in compareifus:
                ii = (nights==night)*(ifuslots==ifu)*(exposures==exp)
                if sum(ii) == 0:
                    compareifus = compareifus[np.where(compareifus!=ifu)]
                    print('{} will not be used.'.format(ifu))

    f = []

    for ifu in ifus:
        faser = []
        wavelength = []
        for fff in ff[np.where((ifuslots==ifu)&(amps==amp))]:
            hdu = fits.open(fff)
            faser.append(hdu['sky_spectrum'].data[45,:])
            wavelength.append(hdu['wavelength'].data[45,:])
            hdu.close()
        f.append(np.array(faser))
    f = np.array(f)
    print('f.shape: ', f.shape)

    fifu, fg = [], []
    for compareifu in compareifus:
        faser2 = []
        wavelength2 = []
        for fff in ff[np.where((ifuslots==compareifu)&(amps==compareamp))]:
            hdu = fits.open(fff)
            faser2.append(hdu['sky_spectrum'].data[45,:])
            wavelength2.append(hdu['wavelength'].data[45,:])
            hdu.close()
        fifu.append(np.array(faser2))
    fifu = np.array(fifu)
    print('fifu.shape: ', fifu.shape)

    return f, fifu

def test():
    ncomp = 8
    shots = ['20180822v008', '20180822v009','20180822v020','20180822v021','20180822v022','20180822v023']
    amp, compareamp = 'RU','LL'
    fs, fifus = get_fibers(shots, amp, compareamp)
    errors, repolys = [],[]

    for f in fs:
        for fifu in fifus:
            f, fmean, fstd = prepare_fiber(f)
            fifu, fifumean, fifustd = prepare_fiber(fifu)
            scprod, fiber_pca = get_scprod(f, ncomp)
            fifunew = get_quasi_fiber(fifu, scprod, fiber_pca, fifumean, fifustd)
            fifu = finish(fifu, fifumean, fifustd)
            re, repoly = get_polyerror(fifu, fifunew, 0)
            errors.append(re)
            repolys.append(repoly)
    errors, repolys = np.array(errors), np.array(repolys)
    return errors, repolys

def main():
    errors, repolys = test()
    plt.figure()
    plt.hist([abs(errors), abs(repolys)], label=['errors','repolys'])
    plt.legend()
    plt.savefig('woclasses.png', bbox_inches='tight')
    plt.close()
    hdu1 = fits.PrimaryHDU(errors)
    hdu2 = fits.ImageHDU(repolys, name='repolys')
    hdul = fits.HDUList([hdu1, hdu2])
    hdul.writeto('PCAdefs.fits', overwrite=True)
    return

main()
