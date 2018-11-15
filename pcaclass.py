import glob
import os
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
#import pickle

class fiber:
    def __init__(self, fiber, ww):
        self.fiber = np.array(fiber) # sky spectra of one fiber at different times
        self.ww = ww # wavelength in this fiber

    def prepare(self):
        self.mean = self.fiber.mean(axis=0) # mean sky spectrum
        self.fiber = self.fiber - self.mean # sklearn pca does this...

    def finish(self):
        self.fiber = self.fiber + self.mean # add it again in the end

class fiber_pca:
    def __init__(self, fiber, ncomp):
        self.fiber = fiber # fiber object
        self.ncomp = ncomp # number of principal components

    def pca(self):
        fiber = self.fiber.fiber # sky spectra
        self.covmat = np.cov(fiber.T) # covariance matrix of the sky spectra: dim 1032x1032
        eigenvals, eigenvecs = np.linalg.eig(self.covmat) # eigenvalues and eigenvectors of covariance matrix: eigenvals show variance in direction of eigenvecs, eigenvecs are the principal components, orthogonal and unit vectors
        eigenvals, eigenvecs = np.real(eigenvals), np.real(eigenvecs) # to remove tiny imaginary parts
        self.eigenpairs = [(np.abs(eigenvals[i]), eigenvecs[:,i]) for i in np.argsort(abs(eigenvals))[::-1]] # sort eigenvalues and eigenvectors by descending order to find the most important components
        imp = self.eigenpairs[:self.ncomp]  # choose ncomp principal components with the highest variance
        imp = [x[1] for x in imp] # we only want the eigenvectors
        self.imp = np.array(imp)
        scprod = self.imp.dot((fiber).T) # projection matrix (scalar product) of the pc's on the input spectra
        self.scprod = np.array([scprod.T[i] / fiber[i].dot(fiber[i]) for i in range(scprod.T.shape[0])]).T # normalize projection
        self.fiber_pca = fiber.dot(self.imp.T) # input spectra in pc base

    def transform(self): # transforms input spectra in pc base back to canonical base (spectra)
        self.fiber_new = np.dot(self.fiber_pca, self.imp) + self.fiber.mean  # new spectrum
        self.fiber.finish() # input spectra

class quasi: # yields quasi eigenvectors (quasi pc's) of another fiber with information of first fiber
    def __init__(self, fiber, scprod, fiber_pca):
        self.fiber = fiber # second fiber
        self.scprod = scprod # projection of pc's on spectra of first fiber
        self.fiber_pca = fiber_pca # projection of spectra of first fiber on pc's of first fiber

    def get_quasi(self):
        fiber = self.fiber.fiber
        quasi = self.scprod.dot(fiber) # quasi principal components as linear combination of second fiber spectra
        norm = np.linalg.norm
        for i in range(quasi.shape[0]):
            quasi[i] = quasi[i]/norm(quasi[i]) # normalize quasi components
        self.quasi = quasi

    def get_new_fiber(self):
        self.fiber_new = np.dot(self.fiber_pca, self.quasi) + self.fiber.mean # new sky spectrum with projection of first fiber spectra on pc's and quasi pc's
        self.fiber.finish() # old sky spectrum

class errors: # fits 5th order polynome to residuals and yields new residuals + some plots
    def __init__(self, fiber, fiber_new, index, wavelength):
        self.fiber = fiber # original sky spectra
        self.fiber_new = fiber_new # new sky spectra
        self.index = index # which spectrum
        self.ww = wavelength[index] # wavelength of spectrum

    def get_errors(self):
        i = self.index
        re = (self.fiber[i] - self.fiber_new[i]) / self.fiber[i] # residuals
        x = np.arange(0, re.shape[0],1)
        sigma = np.nanstd(re) # standard deviation of residuals
        kappa = 3
        flag = np.where((re-re.mean())<kappa*sigma) # ignore high deviations from mean
        self.re = re
        pp = np.polyfit(x[flag],re[flag],5) # fit 5th order polynome to residuals (to continuous offset)
        poly = pp[5]+x*pp[4]+x*x*pp[3]+x*x*x*pp[2]+x*x*x*x*pp[1] + x*x*x*x*x*pp[0]
        self.repoly = re - poly # get rid of continuous offset
        self.fiber_poly = self.fiber_new[i]*(1+poly) # scale new sky spectrum to get rid of the offset

    def fiberplot(self): # plots original sky spectrum, new sky spectrum without poly fit and new sky spectrum with poly fit
        i = self.index
        plt.figure()
        plt.plot(self.ww, self.fiber[i], label='original')
        plt.plot(self.ww, self.fiber_new[i], label = 'no poly')
        plt.plot(self.ww, self.fiber_poly, label='poly')
        plt.xlabel(r'wavelength [$\AA$]')
        plt.ylabel('counts')
        plt.legend()
        plt.show()

    def errorplot(self): # plots residuals before and after poly fit
        plt.figure()
        plt.plot(self.ww, self.re, label='no poly')
        plt.plot(self.ww, self.repoly, label='poly')
        plt.axhline(0, linestyle='--', color = 'grey')
        plt.xlabel(r'wavelength [$\AA$]')
        plt.ylabel('residuals')
        plt.legend()
        plt.show()

    def errorhist(self): # histogram of errors (absolute residuals) before and after poly fit
        plt.figure()
        plt.hist([abs(self.re.flatten()), abs(self.repoly.flatten())], density=True, label = ['no poly', 'poly'])
        plt.legend()
        plt.xlabel('relative error')
        plt.show()

def test(f_pca, f): # compute quasis and errors
    qq = quasi(f, f_pca.scprod, f_pca.fiber_pca)
    qq.get_quasi()
    qq.get_new_fiber()
    err = errors(qq.fiber.fiber, qq.fiber_new, 5, f.ww)
    err.get_errors()
    return err

def plotcompare(errors, label, ifu, amp, compareifu, compareamp, bins=None): # plots histogram of different errors, e.g. null test, different ifu, gauss filtered spectra...
    plt.figure()
    if bins==None:
        plt.hist([abs(x) for x in errors], label=label, normed=True)
    else:
        plt.hist([abs(x) for x in errors], label=label, normed=True, bins=bins)
    plt.legend()
    plt.title('{}_{} to {}_{}'.format(ifu, amp, compareifu, compareamp))
    plt.xlabel('relative error')
    #plt.show()
    plt.savefig('{}_{}_{}_{}.png'.format(ifu, amp, compareifu, compareamp))

def get_fibers1(shots, sigma, ifu, amp, compareifu, compareamp): # gets spectra from hetdex data in maverick (now: fiber, second fiber in another ifu, gauss filtered second fiber)
    night = shots[0][:-4]
    pattern = '/work/03946/hetdex/maverick/red1/reductions/{}/virus/virus0000*/exp0?/virus/multi*.fits'.format(night)

    ff = glob.glob(pattern)
    ff = np.array(ff)

    ifuslots = []
    amps = []
    exposures = []
    allshots = []

    for fff in ff:
        h,t = os.path.split(fff)
        ifuslots.append( t[10:13] )
        amps.append( t[18:20] )
        exposures.append( h.split('/')[-2])
        allshots.append(h.split('/')[-3])

    ifuslots = np.array(ifuslots)
    amps = np.array(amps)
    exposures = np.array(exposures)
    allshots = np.array(allshots)
    ee, ss, aa, shs = np.unique(exposures), np.unique(ifuslots), np.unique(amps), np.unique(shots)

    goodshotmask = np.ones(len(shs), dtype=np.int8)*True

    # only use shots with three exposures, i.e. hetdex shots
    for shot in shots:
        exp = ee[2] # second exposure: 'exp03'
        ii = (exposures == exp) * (allshots == 'virus0000{}'.format(shot[-3:]))
        if not sum(ii) >= 1: # i.e. if there is no third exposure
            goodshotmask[np.where(shs==shot)] = False
            continue
        fin = ff[ii][0]
        print("Reading {}".format(fin))
        try:
            hdu = fits.open(fin)
            a = hdu['sky_spectrum'].data # check if there is a sky spectrum extension
            hdu.close()
        except KeyError:
            hdu.close()
            goodshotmask[np.where(shs==shot)] = False
            print('{} has no sky spectrum'.format(shot))
        shots = shs[np.where(goodshotmask)]
    shots = np.array(shots)
    print('These shots are going to be used: ', shots)

    allgoodshotsmask = np.ones(len(allshots), dtype=np.int8)*False
    allshots = np.array(['{}v{}'.format(night, x[-3:]) for x in allshots])
    for shot in shots:
        allgoodshotsmask[np.where(allshots==shot)] = True

    # these arrays are going to be used instead of dictionary keys
    # only use shots with three exposures and a sky spectrum extension
    ff, allshots, exposures, ifuslots, amps = ff[np.where(allgoodshotsmask)], allshots[np.where(allgoodshotsmask)],exposures[np.where(allgoodshotsmask)], ifuslots[np.where(allgoodshotsmask)], amps[np.where(allgoodshotsmask)]
    ee, ss, aa, shs = np.unique(exposures), np.unique(ifuslots), np.unique(amps), np.unique(shots)
    ifus = ss[:int(floor(ss.shape[0]/4))]
    compareifus = ss[int(floor(3*ss.shape[0]/4)):]

    faser = []
    wavelength = []

    for fff in ff[np.where((ifuslots==ifu)&(amps==amp))]:
        hdu = fits.open(fff)
        faser.append(hdu['sky_spectrum'].data[45,:])
        wavelength.append(hdu['wavelength'].data[45,:])
        hdu.close()
    f = fiber(np.array(faser), np.array(wavelength))
    f.prepare() # first fiber
    #f.append(f0)

    faser2 = []
    wavelength2 = []
    for fff in ff[np.where((ifuslots==compareifu)&(amps==compareamp))]:
        hdu = fits.open(fff)
        faser2.append(hdu['sky_spectrum'].data[45,:])
        wavelength2.append(hdu['wavelength'].data[45,:])
        hdu.close()
    fifu = fiber(np.array(faser2), np.array(wavelength2))
    fifu.prepare() # second fiber
    #fifu.append(fifu0)
    """
    with open('obj/20180822_rel_throughput.pkl', 'rb') as rt:
        a = pickle.load(rt)
    key = list(a.keys())[0]
    key2 = list(a[key].keys())[1]
    rt = a[key][key2]
    rt = np.append(rt, np.ones(22))
    a = None
    frelthrough = np.array(faser)*rt
    frt = fiber(frelthrough, np.array(wavelength))
    frt.prepare()
        fgauss = np.array([gaussian_filter(x[np.isfinite(x)], sigma=sigma) for x in frelthrough])"""
    # gaussian filter of compare ifu
    fgauss = np.array([gaussian_filter(x[np.isfinite(x)], sigma=sigma) for x in faser2])
    fg = fiber(fgauss, np.array(wavelength))
    fg.prepare() # gauss filtered second fiber
    #fg.append(fg0)

    return f, fifu, fg

def get_fibers(shots, sigma, amp, compareamp): # second version of get_fibers : gets lists of first and second fibers
    night = shots[0][:-4]
    pattern = '/work/03946/hetdex/maverick/red1/reductions/{}/virus/virus0000*/exp0?/virus/multi*.fits'.format(night)

    ff = glob.glob(pattern)
    ff = np.array(ff)

    ifuslots = []
    amps = []
    exposures = []
    allshots = []

    for fff in ff:
        h,t = os.path.split(fff)
        ifuslots.append( t[10:13] )
        amps.append( t[18:20] )
        exposures.append( h.split('/')[-2])
        allshots.append(h.split('/')[-3])

    ifuslots = np.array(ifuslots)
    amps = np.array(amps)
    exposures = np.array(exposures)
    allshots = np.array(allshots)
    ee, ss, aa, shs = np.unique(exposures), np.unique(ifuslots), np.unique(amps), np.unique(shots)

    goodshotmask = np.ones(len(shs), dtype=np.int8)*True

        # only use shots with more than one exposure, i.e. hetdex shots
    for shot in shots:
        exp = ee[2] # second exposure: 'exp03'
        ii = (exposures == exp) * (allshots == 'virus0000{}'.format(shot[-3:]))
        if not sum(ii) >= 1: # i.e. if there is no second exposure
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
    allshots = np.array(['{}v{}'.format(night, x[-3:]) for x in allshots])
    for shot in shots:
        allgoodshotsmask[np.where(allshots==shot)] = True

        # these arrays are going to be used instead of dictionary keys
    ff, allshots, exposures, ifuslots, amps = ff[np.where(allgoodshotsmask)], allshots[np.where(allgoodshotsmask)],exposures[np.where(allgoodshotsmask)], ifuslots[np.where(allgoodshotsmask)], amps[np.where(allgoodshotsmask)]
    ee, ss, aa, shs = np.unique(exposures), np.unique(ifuslots), np.unique(amps), np.unique(shots)
    ifus = ss[:int(ss.shape[0]/2)]
    compareifus = ss[int(ss.shape[0]/2):]

    f = []

    for ifu in ifus:
        faser = []
        wavelength = []

        for fff in ff[np.where((ifuslots==ifu)&(amps==amp))]:
            hdu = fits.open(fff)
            faser.append(hdu['sky_spectrum'].data[45,:])
            wavelength.append(hdu['wavelength'].data[45,:])
            hdu.close()
        f0 = fiber(np.array(faser), np.array(wavelength))
        f0.prepare()
        f.append(f0)

    fifu, fg = [], []
    for compareifu in compareifus:
        faser2 = []
        wavelength2 = []
        for fff in ff[np.where((ifuslots==compareifu)&(amps==compareamp))]:
            hdu = fits.open(fff)
            faser2.append(hdu['sky_spectrum'].data[45,:])
            wavelength2.append(hdu['wavelength'].data[45,:])
            hdu.close()
        fifu0 = fiber(np.array(faser2), np.array(wavelength2))
        fifu0.prepare()
        fifu.append(fifu0)
        """
        with open('obj/20180822_rel_throughput.pkl', 'rb') as rt:
            a = pickle.load(rt)
        key = list(a.keys())[0]
        key2 = list(a[key].keys())[1]
        rt = a[key][key2]
        rt = np.append(rt, np.ones(22))
        a = None
        frelthrough = np.array(faser)*rt
        frt = fiber(frelthrough, np.array(wavelength))
        frt.prepare()
            fgauss = np.array([gaussian_filter(x[np.isfinite(x)], sigma=sigma) for x in frelthrough])"""
        # gaussian filter of compare ifu
        fgauss = np.array([gaussian_filter(x[np.isfinite(x)], sigma=sigma) for x in faser2])
        fg0 = fiber(fgauss, np.array(wavelength))
        fg0.prepare()
        fg.append(fg0)

    return f, fifu, fg


def main():
    ncomp = 10 # number of principal components to use
    sigma = 3  # sigma for gaussian filter
    shots = ['20180822v008', '20180822v009','20180822v020','20180822v021','20180822v022','20180822v023']

    amp, compareamp = 'RU', 'LL'
    f, fifu, fg = get_fibers(shots, sigma, amp, compareamp)
    f_pca = []
    errnull, errifu, errgauss = [],[],[]
    j=0
    for ex in f:
        j+=1
        for i in range(len(fifu)):
            f_pca = fiber_pca(ex, ncomp)
            f_pca.pca()
            ernull0 = test(f_pca, ex)
            errifu0 = test(f_pca, fifu[i])
            errgauss0 = test(f_pca, fg[i])
            errnull.append(ernull0.repoly)
            errifu.append(errifu0.repoly)
            errgauss.append(errgauss0.repoly)
            """
            plt.figure(figsize=(20,4))
            plt.plot(fifu[i].ww[errifu0.index], errifu0.repoly, label='residual')
            normed = (fifu[i].fiber[errifu0.index] - np.nanmean(fifu[i].fiber[errifu0.index])) / np.nanstd(fifu[i].fiber[errifu0.index]) * 0.025
            plt.plot(fifu[i].ww[errifu0.index], normed, label='sky')
            plt.xlabel(r'wavelength [$\AA$]')
            plt.axhline(0, linestyle='--', color='grey', alpha=0.5)
            plt.xlim(np.nanmin(fifu[i].ww[errifu0.index]), np.nanmax(fifu[i].ww[errifu0.index]))
            #plt.title('{}_{} {}_{}'.format(ifu, amp, compareifu, compareamp))
            plt.legend()
            plt.savefig('ncomp{}/repolyifu{}{}.png'.format(ncomp, j, i), bbox_inches='tight')
            plt.close()"""
    errnull, errifu, errgauss = np.array(errnull).flatten(), np.array(errifu).flatten(), np.array(errgauss).flatten()
    eins, zwei, drei = abs(errnull),abs(errifu),abs(errgauss)
    plt.figure()
    plt.hist([eins, zwei, drei], label=['null','ifu','gauss'], normed=True, bins = np.arange(0,1,0.1))
    plt.legend()
    plt.title('all errors in 20180822')
    plt.xlabel('relative error')
    #plt.show()
    plt.savefig('pcarepoly20180822.png')
    plt.close()
    plt.figure()
    plt.hist([eins, zwei, drei], label=['null','ifu','gauss'], normed=True, bins = np.arange(0,0.5,0.05))
    plt.legend()
    plt.title('all errors in 20180822')
    plt.xlabel('relative error')
    #plt.show()
    plt.savefig('pcarepoly20180822_1.png')
    plt.close()
    plt.figure()
    plt.hist([eins, zwei, drei], label=['null','ifu','gauss'], normed=True, bins = np.arange(0,0.05,0.005))
    plt.legend()
    plt.title('all errors in 20180822')
    plt.xlabel('relative error')
    #plt.show()
    plt.savefig('pcarepoly20180822_2.png')
    plt.close()

    """
    for ifu in ['025','043','053','077']:
        for compareifu in ['032','076']:

            f, fifu, fg = get_fibers1(shots, sigma, ifu, amp, compareifu, compareamp)

            f_pca = fiber_pca(f, ncomp)
            f_pca.pca()
            #errnull, errifu, errrt, errgauss = test(f_pca, f), test(f_pca, fifu), test(f_pca, frt), test(f_pca, fg)
            errnull = test(f_pca, f)
            errifu = test(f_pca, fifu)
            #errrt = test(f_pca, frt)
            errgauss = test(f_pca, fg)

            plt.figure(figsize=(20,4))
            plt.plot(fifu.ww[errifu.index], errifu.repoly, label='residual')
            normed = (fifu.fiber[errifu.index] - np.nanmean(fifu.fiber[errifu.index])) / np.nanstd(fifu.fiber[errifu.index]) * 0.025
            plt.plot(fifu.ww[errifu.index], normed, label='sky')
            plt.xlabel(r'wavelength [$\AA$]')
            plt.axhline(0, linestyle='--', color='grey', alpha=0.5)
            plt.xlim(np.nanmin(fifu.ww[errifu.index]), np.nanmax(fifu.ww[errifu.index]))
            plt.title('{}_{}_{}_{}'.format(ifu, amp, compareifu, compareamp))
            plt.legend()
            plt.savefig('repoly{}_{}_{}_{}.png'.format(ifu, amp, compareifu, compareamp), bbox_inches='tight')
            #plotcompare([errnull.repoly, errifu.repoly, errgauss.repoly],
            #            label = ['null','ifu', 'gauss'],ifu=ifu, amp=amp, compareifu=compareifu, compareamp=compareamp)"""

main()
