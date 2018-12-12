import numpy as np
import glob
from astropy.io import fits
import os

def main():

    pattern = 'rebinned20180822/rebinned*_exp0?_multi*.fits.fits' # change for the other 2 IFUs
    ff = glob.glob(pattern)
    ff = np.array(ff)
    reihenfolge = np.argsort(ff)
    ff = ff[reihenfolge]

    ifuslots =
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


    #ifu = '042'
    #amp = 'RL'
    for ifu in ss:
        for amp in aa:

            gg = ff[np.where((ifuslots==ifu)&(amps==amp))]
            #print('gg: ', gg)

            allrescaled, allrepoly = [],[]

            fiber = []
            spec=[]
            for f in gg:
                hdu = fits.open(f)
                fiber.append(hdu['sky_spectrum'].data[50,:])
                hdu.close()
            fiber = np.array(fiber)


            fiber = np.concatenate((fiber, fiber[6:]+np.mean(fiber[6:], axis=0)))
            fiber = np.concatenate((fiber, fiber[6:]+2*np.mean(fiber[6:], axis=0)))
            fiber = fiber[6:]

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

            #ifu = '025'
            #amp = 'RL'

            gg = ff[np.where((ifuslots==ifu)&(amps==amp))]
            print('gg: ', gg)

            fiber22 = []
            for f in gg:
                hdu = fits.open(f)
                fiber22.append(hdu['sky_spectrum'].data) #(shape (112,1032))
                hdu.close()
            fiber22 = np.array(fiber22)
            # shape (18, 112, 1032)
            fiber22 = np.transpose(fiber22, [1,0,2])

            for i in range(112):
                fiber2 = fiber22[i]
                """fiber2 = []
                for f in ff:
                    hdu = fits.open(f)
                    fiber2.append(hdu['sky_spectrum'].data[i,:]) # 80 behaves strangely, 20 is better
                    hdu.close()
                fiber2 = np.array(fiber2)"""

                #fiber = np.concatenate((fiber, fiber[6:]+np.mean(fiber[6:], axis=0)))
                fiber2 = np.concatenate((fiber2, fiber2[6:]+np.mean(fiber2[6:], axis=0)))
                #fiber = np.concatenate((fiber, fiber[6:]+2*np.mean(fiber[6:], axis=0)))
                fiber2 = np.concatenate((fiber2, fiber2[6:]+2*np.mean(fiber2[6:], axis=0)))
                #fiber = fiber[6:]
                fiber2 = fiber2[6:]

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
                residuals = (f2-quasitest2std)/f2
                x = np.arange(req2norm.shape[-1])
                for i in range(12): #range(req2norm.shape[0]): bis 12 sind die ursprünglichen Spektren.
                    try:
                        re = residuals[i]
                        sigma = np.nanstd(re[:250])
                        remean = np.nanmean(re[:250])
                        sigma2 = np.nanstd(re[250:])
                        remean2 = np.nanmean(re[250:])
                        kappa = 2.5
                        flag = (np.isfinite(re[:250]))&(abs(re[:250]-remean)<=kappa*sigma)
                        flag2 = (np.isfinite(re[250:]))&(abs(re[250:]-remean2)<=kappa*sigma2)
                        flag = np.append(flag, flag2)
                        pp = np.polyfit(x[np.where(flag)], re[np.where(flag)], 5)
                        poly = pp[5]+x*pp[4]+x*x*pp[3]+x*x*x*pp[2]+x*x*x*x*pp[1] + x*x*x*x*x*pp[0]
                        re = re - poly
                        repoly.append(re)
                        #rescaled[i] = quasitest2std[i]*(1+poly)
                        rescaled.append(quasitest2std[i]*(1+poly))
                    except TypeError:
                        repoly.append(residuals[i])
                        rescaled.append(quasitest2std[i])
                repoly=np.array(repoly)
                allrescaled.append(rescaled)
                allrepoly.append(repoly)

            allrescaled, allrepoly = np.array(allrescaled), np.array(allrepoly)
            allrescaled = np.transpose(allrescaled, [1,0,2])
            allrepoly = np.transpose(allrepoly, [1,0,2])
            print('allrescaled.shape: ', allrescaled.shape)
            print('allrepoly.shape: ', allrepoly.shape)
            print('\nallrepoly = {} +- {}\n'.format(np.nanmean(abs(repoly)), np.nanstd(abs(repoly))))

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


            for i in range(12):
                fin = gg[i+6]
                hdu = fits.open(fin)
                hdu.append(fits.ImageHDU(allrescaled[i], name='pcasky'))
                hdu.append(fits.ImageHDU(allrepoly[i], name='pcarepoly'))
                hdu.writeto(fin, overwrite=True)

main()
