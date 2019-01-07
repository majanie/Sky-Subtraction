import numpy as np
import glob
from astropy.io import fits
import os
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from scipy.signal import medfilt
from scipy.ndimage.filters import gaussian_filter

params = {'legend.fontsize': 'x-large',
         # 'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

def gauss(a,b,c,x):
    return a*np.exp(-(x-b)**2/c)**2

def get_common_sky(nights,ifu):
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
    return common_sky_models, commonexps, commonshots

pattern = '/media/maja/Elements/rebinned20180822/rebinned*_exp0?_multi*.fits.fits' # change for the other 2 IFUs
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
#nights = np.array(nights)
ee, ss, aa, shs = np.unique(exposures), np.unique(ifuslots), np.unique(amps), np.unique(allshots)# np.unique(shots)# np.unique(['{}v{}'.format(night, x[-3:]) for night,x in zip(nights,allshots)])



"""
fiber = []
spec=[]
for f in gg:
    hdu = fits.open(f)
    fiber.append(hdu['sky_spectrum'].data[50,:])
    hdu.close()
fiber = np.array(fiber)

"""

for ifu in ss:#['053']:
    csm, ce, cs = get_common_sky(['20180822'],ifu)

    fiber = csm
    fiber = fiber[6:]
    fiber = np.concatenate((fiber[6:9],fiber[:6], fiber[9:]))
    #print('i hope this works...')
    fiber = np.concatenate((fiber, fiber[3:]+1*np.mean(fiber[3:], axis=0)))
    #fiber = np.concatenate((fiber, gaussian_filter(fiber[6:9],3)-1*np.mean(fiber[9:], axis=0)))

    fiber = np.concatenate((fiber, fiber[3:]+2*np.mean(fiber[3:], axis=0)))

    fiber = np.concatenate((fiber, fiber[3:]+2*np.mean(fiber[3:], axis=0)))

    #fiber = np.concatenate((fiber, 0.5*fiber[6:9]+10*np.mean(fiber[9:18], axis=0)))

    #fiber = np.concatenate((fiber, fiber[9:]+1*np.mean(fiber[9:], axis=0)))
    #print(np.array([gaussian_filter(u,sigma=4) for u in fiber[6:18]]).shape)
    #fiber = np.concatenate((fiber, np.array([gaussian_filter(fiber[i],sigma=4) for i in range(6,18)])))
    #fiber = np.concatenate((fiber, fiber[9:]+2*np.mean(fiber[9:], axis=0)))

    #fiber = np.concatenate((fiber, fiber[6:]+2*np.mean(fiber[6:], axis=0)))

    #fiber[:3] = fiber[:3]+np.array([gauss(10,400,4, np.arange(0,1010,1)) for j in range(3)])

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

    #fiber = np.concatenate((fiber, fiber[:3]+2*np.mean(fiber[3:12], axis=0)))
    #fiber = np.concatenate((fiber, fiber[:3]+1*np.mean(fiber[3:12], axis=0)))

    x = np.array([x/np.linalg.norm(x) for x in fiber])
    scprod = imp.dot(x.T)

    fiber_pca = np.dot(fiber, imp.T)

    for amp in aa: #['LL']:
        allrescaled, allrepoly = [],[]
        #ifu = '025'
        #amp = 'RU'

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
        spectrum2 = spectrum2[6:]
        fiber22 = fiber22[6:]

        fiber22 = np.concatenate((fiber22[6:9], fiber22[:6], fiber22[9:]))
        spectrum2 = np.concatenate((spectrum2[6:9], spectrum2[:6], spectrum2[9:]))

        fiber22[:3] = fiber22[:3]+np.array([[gauss(20,600,8, np.arange(0,1032,1)) for i in range(112)] for j in range(3)])
        spectrum2[:3] = spectrum2[:3]+np.array([[gauss(20,600,8, np.arange(0,1032,1)) for i in range(112)] for j in range(3)])

        fiber22 = np.transpose(fiber22, [1,0,2])
        spectrum2 = np.transpose(spectrum2, [1,0,2])
        #print('fiber22: ', fiber22.shape)

        for i in range(112):
            fiber2 = fiber22[i]

            """fiber2 = []
            for f in ff:
                hdu = fits.open(f)
                fiber2.append(hdu['sky_spectrum'].data[i,:]) # 80 behaves strangely, 20 is better
                hdu.close()
            fiber2 = np.array(fiber2)"""

            fiber2 = np.concatenate((fiber2, fiber2[3:]+1*np.mean(fiber2[3:], axis=0)))
            #fiber2 = np.concatenate((fiber2, gaussian_filter(fiber2[:3],3)-1*np.mean(fiber2[3:], axis=0)))

            fiber2 = np.concatenate((fiber2, fiber2[3:]+2*np.mean(fiber2[3:], axis=0)))
            fiber2 = np.concatenate((fiber2, fiber2[3:]+2*np.mean(fiber2[3:], axis=0)))
            #fiber2 = np.concatenate((fiber2, fiber2[:3]+2*np.mean(fiber2[3:12], axis=0)))
            #fiber2 = np.concatenate((fiber2, fiber2[:3]+1*np.mean(fiber2[3:12], axis=0)))

            #fiber2 = np.concatenate((fiber2, 0.5*fiber2[:3]+10*np.mean(fiber2[3:12], axis=0)))

            #fiber2 = np.concatenate((fiber2, fiber2[3:]+1*np.mean(fiber2[3:], axis=0)))
            #fiber2 = np.concatenate((fiber2, np.array([gaussian_filter(u,sigma=4) for u in fiber2[:12]])))
            #fiber2 = np.concatenate((fiber2, fiber2[3:]+2*np.mean(fiber2[3:], axis=0)))

            #fiber = np.concatenate((fiber, fiber[6:]+2*np.mean(fiber[6:], axis=0)))
            #fiber2 = np.concatenate((fiber2, fiber2[6:]+1*np.mean(fiber2[6:], axis=0)))
            #fiber2 = np.concatenate((fiber2, fiber2[6:]+2*np.mean(fiber2[6:], axis=0)))
            #fiber2 = fiber2[6:]
            #mean1 = fiber.mean(axis=0)
            mean2 = fiber2.mean(axis=0)
            #std1 = fiber.std(axis=0)
            std2 = fiber2.std(axis=0)

            #fiber = (fiber - mean1)/std1
            fiber2 = (fiber2 - mean2) / std2

            """cov_mat = np.cov(fiber.T)

            eigenvals, eigenvecs = np.linalg.eig(cov_mat)
            eigenvals = np.real(eigenvals)
            eigenvecs = np.real(eigenvecs)

            eigenpairs = [(np.abs(eigenvals[i]), eigenvecs[:,i]) for i in np.argsort(abs(eigenvals))[::-1]]

            ncomp = 9 # 15 und 20 bei (18,112,1032) machen keinen großen, nur kleinen Unterschied
            imp =  eigenpairs[:ncomp]
            imp = [x[1] for x in imp]
            imp = np.array(imp)

            x = np.array([x/np.linalg.norm(x) for x in fiber])
            scprod = imp.dot(x.T)

            fiber_pca = np.dot(fiber, imp.T)"""

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

        for i in range(12):
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
        """plt.figure(figsize=(20,8))
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
