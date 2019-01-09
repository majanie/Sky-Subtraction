import numpy as np
import glob
from astropy.io import fits
import os
#import matplotlib.pyplot as plt
#import matplotlib.pylab as pylab
from scipy.signal import medfilt
from scipy.ndimage.filters import gaussian_filter

# this function will create a linearely binned version
# of a particular spectrum
from srebin import linlin

#p))arams = {'legend.fontsize': 'x-large',
         # 'figure.figsize': (15, 5),
#         'axes.labelsize': 'x-large',
#         'axes.titlesize':'x-large',
#         'xtick.labelsize':'x-large',
#         'ytick.labelsize':'x-large'}
#pylab.rcParams.update(params)

def rebin(ff):

    """ turnes sky spectrum, spectrum, and fiber-to-fiber arrays into linearly binned versions """
    number = ff.shape[0]
    i = 0

    sky_spectra, spectra, fiber_to_fiber = [], [], []
    for fin in ff:
        i += 1
        ww, rebinned = get_rebinned(fin)
        print('rebinned {}/{}'.format(i, number))
        sky_spectra.append(rebinned['sky_spectrum'])
        spectra.append(rebinned['spectrum'])
        fiber_to_fiber.append(rebinned['fiber_to_fiber'])
    ww = ww
    sky_spectra = np.array(sky_spectra)
    spectra = np.array(spectra)
    fiber_to_fiber = np.array(fiber_to_fiber)

    return sky_spectra, spectra, fiber_to_fiber, ww

def gauss(a,b,c,x):
    return a*np.exp(-(x-b)**2/c)**2

def get_skymodels(sky_spectra, fiber_to_fiber):

    """ get (mean) sky for each amplifier """
    sky_models = []
    for i in range(sky_spectra.shape[0]):
        sky_model = np.nanmedian( sky_spectra[i] / fiber_to_fiber[i], axis = 0)
        sky_models.append(sky_model)
    sky_models = np.array(sky_models)
    #print('sky_models shape: ', sky_models.shape)
    return sky_models

def get_commonsky(sky_models, shots, allshots, exposures, niceifus, ifuslots):

    """ get mean sky for each shot, ifu and exposure """

    commonskies = sky_models#np.ones(shape = sky_models.shape) # try ones?
    for shot in shots:
        for exp in ['exp01','exp02','exp03']:
            for ifu in np.unique(ifuslots):
                thisnightandexp = np.where((allshots==shot)&(exposures==exp)&(ifuslots==ifu))
                print('thisnightandexp', thisnightandexp)
                commonsky = np.nanmedian( sky_models[thisnightandexp] , axis = 0)
                commonskies[thisnightandexp] = commonsky
    return commonskies

"""def get_common_sky(nights,ifu):
    patterns = ['/media/maja/Elements/rebinned{}/rebinned*_exp0?_multi*.fits'.format(night) for night in nights]
    ff = np.array([])
    for pattern in patterns:
        ff = np.append(ff, np.array(glob.glob(pattern))) # get list of multifits files of nights
    ifuslots = []
    amps = []
    exposures = []
    allshots = []

    for fff in ff: # list of all ifuslots etc for all multifits files in tha same order as ff
                    # rebinned20180113v013_exp02_multi_051_105_051_RL.fits.fits
        h,t = os.path.split(fff)
        ifuslots.append( t[37:40] )
        amps.append( t[45:47] )
        exposures.append( t[21:26])
        allshots.append(t[8:20])

    ifuslots = np.array(ifuslots)
    amps = np.array(amps)
    exposures = np.array(exposures)
    allshots = np.array(allshots)
    nights = np.array(nights)
    ee, ss, aa, shs = np.unique(exposures), np.unique(ifuslots), np.unique(amps), np.unique(allshots)# np.unique(shots)# np.unique(['{}v{}'.format(night, x[-3:]) for night,x in zip(nights,allshots)])

    hier = np.where(ifuslots==ifu)
    ifuslots = ifuslots[hier]
    amps = amps[hier]
    exposures = exposures[hier]
    allshots = allshots[hier]
    ff = ff[hier] # !!!
    sky_spectra, fiber_to_fiber = [],[]
    for i in range(ff.shape[0]):
        #print('Reading {}...'.format(ff[i]))
        a = fits.open(ff[i])
        sky_spectra.append(a['rebinned_sky'].data)
        fiber_to_fiber.append(a['rebinned_fiber_to_fiber'].data)
        a.close()
    sky_spectra, fiber_to_fiber = np.array(sky_spectra), np.array(fiber_to_fiber) # shape (i, 110, 1010) oder so
    print('sky_spectra.shape: ', sky_spectra.shape)
    #print('fiber_to_fiber.shape: ', fiber_to_fiber.shape)
    print('THIS: ', sky_spectra[np.where((allshots=='20180822v022')&(amps=='LL'))].shape)  # Here we add the ''diffuse Lya emission''
    sky_spectra[np.where((allshots=='20180822v022')&(amps=='LL'))] = sky_spectra[np.where((allshots=='20180822v022')&(amps=='LL'))]+np.array([[gauss(20,600,8, np.arange(0,1010,1)) for i in range(112)] for j in range(3)])

    #sky_models = []
    #for i in range(ff.shape[0]):
    #   sky_models.append(np.nanmean(sky_spectra[i]/fiber_to_fiber[i], axis = 0))
    sky_models = np.nanmean(sky_spectra/fiber_to_fiber, axis=1)
    #print('sky_models.shape: ', sky_models.shape)
    common_sky_models = []
    commonexps, commonshots = [],[]
    for shot in shs:
        for exp in ee:
            here = np.where((allshots==shot)&(exposures==exp)&(ifuslots==ifu))
            theseskymodels = sky_models[here]
            #print('theseskymodels.shape: ', theseskymodels.shape)
            csm = np.nanmean(theseskymodels, axis=0)
            #print('csm.shape: ', csm.shape)
            common_sky_models.append(csm)
            commonexps.append(exp)
            commonshots.append(shot)
    common_sky_models = np.array(common_sky_models)
    commonexps, commonshots = np.array(commonexps), np.array(commonshots)
    #print('common_sky_models.shape: ', common_sky_models.shape)
    return common_sky_models, commonexps, commonshots"""

def find_indices(shots):
    # assumes that only shots from one night are used... not anymore
    nights = [shot[:-4] for shot in shots]
    pattern = ['/work/03946/hetdex/maverick/red1/reductions/{}/virus/virus0000*/exp0?/virus/multi*.fits'.format(night) for night in nights]

    ff = np.concatenate([glob.glob(pp) for pp in pattern])
    print('ff: ',ff)
    ff = np.array(ff)

    ifuslots = []
    amps = []
    exposures = []
    allshots = []
    allnights = []

    for f in ff:
        h,t = os.path.split(f)
        ifuslots.append( t[10:13] )
        amps.append( t[18:20] )
        exposures.append( h.split('/')[-2])
        allshots.append(h.split('/')[-3])
        allnights.append(h.split('/')[-5])

    ifuslots = np.array(ifuslots)
    amps = np.array(amps)
    exposures = np.array(exposures)
    allshots = np.array(allshots)
    allnights = np.array(allnights)
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
    allshots = np.array(['{}v{}'.format(night, x[-3:]) for night, x in zip(allnights,allshots)])
    for shot in shots:
        allgoodshotsmask[np.where(allshots==shot)] = True

    # these arrays are going to be used instead of dictionary keys
    ff, allshots, exposures, ifuslots, amps = ff[np.where(allgoodshotsmask)], allshots[np.where(allgoodshotsmask)],exposures[np.where(allgoodshotsmask)], ifuslots[np.where(allgoodshotsmask)], amps[np.where(allgoodshotsmask)]

    return ff, allshots, exposures, ifuslots, amps

def get_rebinned(fin ,extensions=['spectrum', 'sky_spectrum', 'fiber_to_fiber'], start = 3494.74, step =  1.9858398, stop = 5500.):

    #print("Reading {}".format(fin))
    hdu = fits.open(fin)

    wl = hdu['wavelength'].data

    #start,stop = 3503.9716796, 5396.477
    N = int( np.ceil( (stop - start)/step ) )

    rebinned = {}
    for ext in extensions:
        #for j in range(hdu[ext].data.shape[1]): # This might cause big errors...
            #isnull = np.unique(hdu[ext].data[:,j])
            #isnull = isnull[np.where(isnull != 0)]
            #if len(isnull)==0:
            #    hdu[ext].data[:, j] = np.ones(hdu[ext].data.shape[0])*np.nan
        #print("Rebinning {}".format(ext))

        new = np.zeros([wl.shape[0], N])
        hduextdata = hdu[ext].data
        for i in range(wl.shape[0]):
            w = wl[i,:]
            f = hduextdata[i,:]
            start = start
            step =  step
            stop = stop
            lw, lf = linlin(w, f, start, step, stop, dowarn=False)
            #lw = np.arange(start, stop, step)
            #lf = model_resampled_10A = spectres(lw, w, f)

            # hack as they may not all necessareyly have the same length
            new[i,:min(N, len(lf))] = lf[:min(N, len(lf))]

        rebinned[ext] = new
    return lw, rebinned

"""pattern = '/media/maja/Elements/rebinned20180822/rebinned*_exp0?_multi*.fits.fits' # change for the other 2 IFUs
ff = glob.glob(pattern)
ff = np.array(ff)
reihenfolge = np.argsort(ff)
ff = ff[reihenfolge]

ifuslots = []
amps = []
exposures = []
allshots = []

for fff in ff: # list of all ifuslots etc for all multifits files in tha same order as ff
                # rebinned20180113v013_exp02_multi_051_105_051_RL.fits.fits
    h,t = os.path.split(fff)
    ifuslots.append( t[37:40] )
    amps.append( t[45:47] )
    exposures.append( t[21:26])
    allshots.append(t[8:20])

ifuslots = np.array(ifuslots)
amps = np.array(amps)
exposures = np.array(exposures)
allshots = np.array(allshots)
#nights = np.array(nights)"""
#ee, ss, aa, shs = np.unique(exposures), np.unique(ifuslots), np.unique(amps), np.unique(allshots)# np.unique(shots)# np.unique(['{}v{}'.format(night, x[-3:]) for night,x in zip(nights,allshots)])



"""
fiber = []
spec=[]
for f in gg:
    hdu = fits.open(f)
    fiber.append(hdu['sky_spectrum'].data[50,:])
    hdu.close()
fiber = np.array(fiber)

"""
def flag_ifus(flagged_ifus, ifuslots, allshots):

    """ niceifus is a boolean array that indicates which ifus are not flagged (without stars) """

    niceifus = np.ones(shape=ifuslots.shape, dtype=np.int8)*True
    for key in flagged_ifus.keys():
        for flagged_ifu in flagged_ifus[key]:
            niceifus[np.where((ifuslots==flagged_ifu)*(allshots == key))] = False
    print('niceifus',niceifus)
    print('np.unique(ifuslots[niceifus])',np.unique(ifuslots[niceifus]))
    print('np.unique(ifuslots[np.where(niceifus)])',np.unique(ifuslots[np.where(niceifus)]))
    return niceifus

def main():
    shots = ['20180124v010', '20180822v022']
    ff, allshots, exposures, ifuslots, amps = find_indices(shots)
    ee, ss, aa, shs = np.unique(exposures), np.unique(ifuslots), np.unique(amps), np.unique(allshots)# np.unique(shots)# np.unique(['{}v{}'.format(night, x[-3:]) for night,x in zip(nights,allshots)])

    flagged_ifus = {}
    overwrite = False
    niceifus = flag_ifus(flagged_ifus, ifuslots, allshots)
    print('flagged ifus')
    sky_spectra, spectra, fiber_to_fiber, ww = rebin(ff)
    print('rebinned')
    sky_models = get_skymodels(sky_spectra, fiber_to_fiber)
    print('sky_models')
    commonskies = get_commonsky(sky_models, shots, allshots, exposures, niceifus, ifuslots)
    print('common sky')


    for ifu in ss:#['053']:
        #csm, ce, cs = get_common_sky() # (['20180822'],ifu)
        csm = commonskies[np.where(ifuslots==ifu)]
        fiber = csm
        #fiber = fiber[6:]
        #fiber = np.concatenate((fiber[6:9],fiber[:6], fiber[9:]))
        #print('i hope this works...')
        fiber = np.concatenate((fiber, fiber[:]+1*np.mean(fiber[:], axis=0)))
        #fiber = np.concatenate((fiber, gaussian_filter(fiber[6:9],3)-1*np.mean(fiber[9:], axis=0)))

        fiber = np.concatenate((fiber, fiber[:]+2*np.mean(fiber[:], axis=0)))
        fiber = np.concatenate((fiber, fiber[:]+2*np.mean(fiber[:], axis=0)))

        mean1 = fiber.mean(axis=0)
        std1 = fiber.std(axis=0)

        fiber = (fiber - mean1)/std1

        cov_mat = np.cov(fiber.T)

        eigenvals, eigenvecs = np.linalg.eig(cov_mat)
        eigenvals = np.real(eigenvals)
        eigenvecs = np.real(eigenvecs)

        eigenpairs = [(np.abs(eigenvals[i]), eigenvecs[:,i]) for i in np.argsort(abs(eigenvals))[::-1]]

        ncomp = 12 # 15 und 20 bei (18,112,1032) machen keinen großen, nur kleinen Unterschied
        imp =  eigenpairs[:ncomp]
        imp = [x[1] for x in imp]
        imp = np.array(imp)

        x = np.array([x/np.linalg.norm(x) for x in fiber])
        scprod = imp.dot(x.T)

        fiber_pca = np.dot(fiber, imp.T)

        for amp in aa: #['LL']:
            allrescaled, allrepoly = [],[]

            gg = ff[np.where((ifuslots==ifu)&(amps==amp))]
            #print('gg: ', gg)

            fiber22 = []
            spectrum2 = []
            for f in gg:
                hdu = fits.open(f)
                fiber22.append(hdu['sky_spectrum'].data) #(shape (112,1032))
                spectrum2.append(hdu['spectrum'].data)
                hdu.close()
            fiber22 = np.array(fiber22)
            # shape (18, 112, 1032)
            spectrum = np.array(spectrum2)

            fiber22 = np.transpose(fiber22, [1,0,2])
            spectrum2 = np.transpose(spectrum2, [1,0,2])
            #print('fiber22: ', fiber22.shape)

            for i in range(112):
                fiber2 = fiber22[i]

                fiber2 = np.concatenate((fiber2, fiber2[:]+1*np.mean(fiber2[:], axis=0)))

                fiber2 = np.concatenate((fiber2, fiber2[:]+2*np.mean(fiber2[:], axis=0)))
                fiber2 = np.concatenate((fiber2, fiber2[:]+2*np.mean(fiber2[:], axis=0)))
                mean2 = fiber2.mean(axis=0)
                std2 = fiber2.std(axis=0)

                fiber2 = (fiber2 - mean2) / std2

                y = np.array([faser/np.linalg.norm(faser) for faser in fiber2])

                quasi2 = np.dot(scprod, y)

                quasi2std, quasi2norm = np.zeros(quasi2.shape),np.zeros(quasi2.shape)
                for i in range(quasi2.shape[0]):
                    quasi2norm[i] = quasi2[i]/ np.linalg.norm(quasi2[i])
                    quasi2std[i] = (quasi2[i] - np.mean(quasi2[i]))/np.std(quasi2[i])*np.std(imp[i])+np.mean(imp[i])

                quasitest2std = np.dot(fiber_pca, quasi2std)
                quasitest2norm = np.dot(fiber_pca, quasi2norm)

                for i in range(quasitest2std.shape[0]):
                    quasitest2std[i] = quasitest2std[i]*std2 + mean2
                    quasitest2norm[i] = quasitest2norm[i]*std2 + mean2

                fiber2n = np.ones(fiber2.shape)
                for i in range(len(fiber2)):
                    fiber2n[i] = fiber2[i]*std2+mean2

                req2std = (fiber2n-quasitest2std)/(fiber2n)
                req2norm = (fiber2n-quasitest2norm)/(fiber2n)

                quasitest2norm = np.dot(fiber_pca, quasi2norm)*std2 + mean2
                #rescaled = np.zeros(shape=quasitest2norm.shape)
                repoly=[]
                rescaled = []
                f2 = fiber2*std2+mean2
                #residuals = (f2-quasitest2std)/f2
                residuals = (f2 - quasitest2std)/f2
                x = np.arange(req2norm.shape[-1])
                for i in range(12): #range(req2norm.shape[0]): bis 12 sind die ursprünglichen Spektren.
                    re = residuals[i]
                    sigma = np.nanstd(re[:250])
                    remean = np.nanmean(re[:250])
                    sigma2 = np.nanstd(re[250:])
                    remean2 = np.nanmean(re[250:])
                    kappa = 1
                    flag = (np.isfinite(re[:250]))&(abs(re[:250]-remean)<=kappa*sigma)
                    flag2 = (np.isfinite(re[250:]))&(abs(re[250:]-remean2)<=kappa*sigma2)
                    flag = np.append(flag, flag2)
                    pp = np.polyfit(x[np.where(flag)], re[np.where(flag)], 5)
                    poly = pp[5]+x*pp[4]+x*x*pp[3]+x*x*x*pp[2]+x*x*x*x*pp[1] + x*x*x*x*x*pp[0]
                    re = re - poly
                    repoly.append(re)
                    #rescaled[i] = quasitest2std[i]*(1+poly)
                    rescaled.append(quasitest2std[i]*(1+poly))
                repoly=np.array(repoly)
                allrescaled.append(rescaled)
                allrepoly.append(repoly)

            allrescaled, allrepoly = np.array(allrescaled), np.array(allrepoly)
            allrescaled = np.transpose(allrescaled, [1,0,2])
            allrepoly = np.transpose(allrepoly, [1,0,2])
            #print('allrescaled.shape: ', allrescaled.shape)
            #print('allrepoly.shape: ', allrepoly.shape)
            #print('\nallrepoly = {} +- {}'.format(np.nanmean(abs(repoly)), np.nanstd(abs(repoly))))
            print('\nallrepoly = {}'.format(np.nanstd(np.append(allrepoly[0,:,250:580], allrepoly[0,:,620:]))) )
            print('\nallrepoly = {}'.format(np.nanstd(np.append(allrepoly[1,:,250:580], allrepoly[1,:,620:]))))
            print('\nallrepoly = {}'.format(np.nanstd(np.append(allrepoly[2,:,250:580], allrepoly[2,:,620:]))))



            """for i in range(12):
                fin = gg[i+6]  # this is specfic for the 20180822 shots... CHANGE IT!!!
                hdu = fits.open(fin)
                hdu.append(fits.ImageHDU(allrescaled[i], name='pcasky'))
                hdu.append(fits.ImageHDU(allrepoly[i], name='pcarepoly'))
                hdu.writeto(fin, overwrite=True)
                print('Wrote {}'.format(fin))

            #print(allrescaled.shape)
            plt.figure(figsize=(20,8))
            ax1 = plt.subplot(421)
            plt.plot(fiber22[35,0,580:620]-allrescaled[0,35,580:620], label='new')
            #plt.plot(fiber22[55,0,580:620], label='with Lya')
            plt.plot((gauss(20,600,8,  np.arange(0,1032,1)))[580:620], linestyle=':', label='without Lya')
            plt.legend()
            plt.subplot(423)
            plt.plot(fiber22[35,1,580:620]-allrescaled[1,35,580:620])
            #plt.plot(fiber22[55,1,580:620])
            plt.plot((gauss(20,600,8, np.arange(0,1032,1)))[580:620], linestyle=':')
            plt.subplot(425)
            plt.plot(fiber22[35,2,580:620]-allrescaled[2,35,580:620])
            #plt.plot(fiber22[55,2,580:620])
            plt.plot((gauss(20,600,8,  np.arange(0,1032,1)))[580:620], linestyle=':')
            plt.subplot(427)
            plt.plot(fiber22[35,5,580:620]-allrescaled[5,35,580:620])
            ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.35),
              fancybox=True, shadow=False, ncol=3)
            plt.subplot(422)
            plt.plot(allrepoly[0,35])
            plt.axhline(-0.01, linestyle=':', color='grey')
            plt.axhline(0.01, linestyle=':', color='grey')
            plt.axhline(0, linestyle=':', color='grey')
            plt.ylim(-0.03, 0.03)
            plt.subplot(424)
            plt.plot(allrepoly[1,35])
            plt.axhline(-0.01, linestyle=':', color='grey')
            plt.axhline(0.01, linestyle=':', color='grey')
            plt.axhline(0, linestyle=':', color='grey')
            plt.ylim(-0.03, 0.03)
            plt.subplot(426)
            plt.plot(allrepoly[2,35])
            plt.axhline(-0.01, linestyle=':', color='grey')
            plt.axhline(0.01, linestyle=':', color='grey')
            plt.axhline(0, linestyle=':', color='grey')
            plt.ylim(-0.03, 0.03)
            plt.subplot(428)
            plt.plot(allrepoly[5,35])
            plt.axhline(-0.01, linestyle=':', color='grey')
            plt.axhline(0.01, linestyle=':', color='grey')
            plt.axhline(0, linestyle=':', color='grey')
            plt.ylim(-0.03, 0.03)
            plt.show()

            plt.figure()
            plt.hist([abs(allrepoly[0,:, :250]).flatten(),abs(allrepoly[0,:,250:580]).flatten(),abs(allrepoly[0,:,620:800]).flatten(),abs(allrepoly[0,:,800:]).flatten()], bins=np.arange(0, 0.1, 0.01), density=True)
            plt.show()
            plt.figure(figsize=(20,8))
            ax1 = plt.subplot(321)
            plt.plot(spectrum2[35,0,580:620]-allrescaled[0,35,580:620], label='new')
            plt.plot(spectrum2[35,0, 580:620]-fiber22[35,0,580:620], label='with Lya')
            plt.plot((gauss(20,600,8, np.arange(0,1032,1)))[580:620], linestyle=':', label='without Lya')
            plt.legend()
            plt.subplot(323)
            plt.plot(spectrum2[35,1,580:620]-allrescaled[1,35,580:620], label='new')
            plt.plot(spectrum2[35,1, 580:620]-fiber22[35,1,580:620], label='with Lya')
            plt.plot((gauss(20,600,8, np.arange(0,1032,1)))[580:620], linestyle=':', label='without Lya')
            plt.subplot(325)
            plt.plot(spectrum2[35,2,580:620]-allrescaled[2,35,580:620], label='new')
            plt.plot(spectrum2[35,2, 580:620]-fiber22[35,2,580:620], label='with Lya')
            plt.plot((gauss(20,600,8, np.arange(0,1032,1)))[580:620], linestyle=':', label='without Lya')
            ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.21),
              fancybox=True, shadow=False, ncol=3)
            plt.subplot(322)
            plt.plot(spectrum2[55,6]-fiber22[55,6], label='new')
            plt.plot(spectrum2[55,6]-allrescaled[6,55], label='new')
            plt.subplot(324)
            plt.plot(spectrum2[55,7]-fiber22[55,7], label='new')
            plt.plot(spectrum2[55,7]-allrescaled[7,55], label='new')
            plt.subplot(326)
            plt.plot(spectrum2[55,8])#-fiber22[55,2], label='new')
            plt.plot(fiber22[55, 8])
            #plt.plot(spectrum2[55,2]-allrescaled[2,55], label='new')
            plt.show()"""
    """
    for i in range(len(gg)):
        fin = gg[i]
        hdu = fits.open(fin)
        try:
            del hdu['pcasky']
            del hdu['pcarepoly']
            #hdu.append(fits.ImageHDU(allrescaled[i], name='pcasky'))
            #hdu.append(fits.ImageHDU(allrepoly[i], name='pcarepoly'))
            hdu.writeto(fin, overwrite=True)
        except:
            pass
        hdu.close()

    """
main()
