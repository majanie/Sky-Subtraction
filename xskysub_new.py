import sys
import os
import glob
import numpy as np
from astropy.io import fits

# this function will create a linearely binned version
# of a particular spectrum
from srebin import linlin

def flag_ifus(flagged_ifus, ifuslots, allshots):

    """ niceifus is a boolean array that indicates which ifus are not flagged (without stars) """

    niceifus = np.ones(shape=ifuslots.shape, dtype=np.int8)*True
    for key in flagged_ifus.keys():
        for flagged_ifu in flagged_ifus[key]:
            niceifus[(ifuslots==flagged_ifu)*(allshots == key)] = False
    #print(np.unique(ifuslots[np.where(niceifus)]))
    return niceifus

def rebin(ff, allshots, exposures, ifuslots, amps):

    """ turnes sky spectrum, spectrum, and fiber-to-fiber arrays into linearly binned versions """
    number = ff.shape[0]
    #i = -1
    FORCE_REBIN = False

    sky_spectra, spectra, fiber_to_fiber = [], [], []
    sky_subtracted = []
    for i in range(ff.shape[0]):
        fin = ff[i]
        if FORCE_REBIN:
            ww, rebinned = get_rebinned(fin)
            if (i%100)==0:
                print('rebinned {}/{}'.format(i, number))
            sky_spectra.append(rebinned['sky_spectrum'])
            spectra.append(rebinned['spectrum'])
            fiber_to_fiber.append(rebinned['fiber_to_fiber'])
            hdu1 = fits.open(fin)
            #hdu1["spectrum"], 
            hdu = fits.HDUList([fits.PrimaryHDU(hdu1["spectrum"].data), hdu1["sky_spectrum"] ])
            #hdu.append(fits.ImageHDU( xskysub[i], name='xsky_subtracted'))
            #hdu.append(fits.ImageHDU( res[i], name='rel_error'))
            # spectra, xsky
            hdu.append(fits.ImageHDU(rebinned['spectrum'], name='spectrum_rb'))
            #hdu.append(fits.ImageHDU(xsky[i], name='xsky_spectrum_rb'))
            hdu.append(fits.ImageHDU(rebinned['sky_spectrum'], name='sky_spectrum_rb'))

            hdu.writeto('../xsky/{}/{}/x_{}'.format(allshots[i], exposures[i], fin.split('/')[-1]), overwrite=True)
            hdu.close()
            print('wrote to ../xsky/{}/{}/x_{}'.format(allshots[i], exposures[i], fin.split('/')[-1]))
        else:
            try:
                hdu = fits.open('../xsky/{}/{}/x_{}'.format(allshots[i], exposures[i], fin.split('/')[-1]))
                sky_spectra.append(hdu["sky_spectrum_rb"].data)
                spectra.append(hdu["spectrum_rb"].data)
                fiber_to_fiber.append(hdu["fiber_to_fiber_rb"].data)
                sky_subtracted.append(hdu["sky_subtracted_rb"].data)
                print("found ../xsky/{}/{}/x_{}".format(allshots[i], exposures[i], fin.split('/')[-1]))
            except:
                ww, rebinned = get_rebinned(fin)
                if (i%100)==0:
                    print('rebinned {}/{}'.format(i, number))
                sky_spectra.append(rebinned['sky_spectrum'])
                spectra.append(rebinned['spectrum'])
                fiber_to_fiber.append(rebinned['fiber_to_fiber'])
                hdu1 = fits.open(fin)
                #hdu1["spectrum"], 
                hdu = fits.HDUList([fits.PrimaryHDU(hdu1["spectrum"].data), hdu1["sky_spectrum"] ])
                #hdu.append(fits.ImageHDU( xskysub[i], name='xsky_subtracted'))
                #hdu.append(fits.ImageHDU( res[i], name='rel_error'))
                # spectra, xsky
                hdu.append(fits.ImageHDU(rebinned['spectrum'], name='spectrum_rb'))
                #hdu.append(fits.ImageHDU(xsky[i], name='xsky_spectrum_rb'))
                hdu.append(fits.ImageHDU(rebinned['sky_spectrum'], name='sky_spectrum_rb'))
                hdu.append(fits.ImageHDU(rebinned["fiber_to_fiber"], name="fiber_to_fiber_rb"))
                hdu.writeto('../xsky/{}/{}/x_{}'.format(allshots[i], exposures[i], fin.split('/')[-1]), overwrite=True)
                hdu.close()
                print('wrote to ../xsky/{}/{}/x_{}'.format(allshots[i], exposures[i], fin.split('/')[-1]))
    ww = np.arange(0,1010,1)
    sky_spectra = np.array(sky_spectra)
    spectra = np.array(spectra)
    fiber_to_fiber = np.array(fiber_to_fiber)

    return sky_spectra, spectra, fiber_to_fiber, ww

def get_skymodels(sky_spectra, fiber_to_fiber):

    """ get (mean) sky for each amplifier """
    sky_models = []
    for i in range(sky_spectra.shape[0]):
        sky_model = np.nanmedian( sky_spectra[i] / fiber_to_fiber[i], axis = 0)
        sky_models.append(sky_model)
    sky_models = np.array(sky_models)
    #print('sky_models shape: ', sky_models.shape)
    return sky_models

def get_commonsky(sky_models, shots, allshots, exposures, niceifus):

    """ get mean sky for each shot and exposure """

    commonskies = np.ones(shape = sky_models.shape) # try ones?
    for shot in shots:
        for exp in ['exp01','exp02','exp03']:
            thisnightandexp = np.where((allshots==shot)&(exposures==exp)&(niceifus))
            commonsky = np.nanmedian( sky_models[thisnightandexp] , axis = 0)
            commonskies[thisnightandexp] = commonsky
    return commonskies

def get_rel_throughput(sky_models, commonskies):

    """ get relative throughput of each amplifier """

    rel_throughput = []#np.zeros(shape = sky_models.shape)
    for i in range(sky_models.shape[0]):
        rel_throughput.append(sky_models[i] / commonskies[i])
    rel_throughput = np.array(rel_throughput)
    return rel_throughput

def get_xrel_throughput(rel_throughput, allshots, exposures, ifuslots, amps, niceifus):

    """ get mean relative throughput without using flagged ifus and the shot itself """

    xrel_throughput = []#np.zeros(shape = rel_throughput.shape) # try ones instead of zeros?
    for i in range(rel_throughput.shape[0]):
        shot, exp, ifu, amp = allshots[i], exposures[i], ifuslots[i], amps[i]
        here = (exposures==exp)&(ifuslots==ifu)&(amps==amp)&(niceifus) #(allshots!=shot)&

        if sum(here) == 0:
            print('No xrt could be computed for shot {} exp {} ifu {} amp {}.'.format(shot, exp, ifu, amp))
            xrel_throughput.append(rel_throughput[i])
        else:
            xrelthrough = np.nanmedian(rel_throughput[np.where(here)], axis = 0)
            xrel_throughput.append(xrelthrough)
    xrel_throughput = np.array(xrel_throughput)
    return xrel_throughput

def get_xskyspectra(commonskies, xrel_throughput, fiber_to_fiber):

    """ the xsky spectrum is the common sky times mean rel throughput times fiber-to-fiber array
        and has the same shape as the rebinned sky spectrum """

    xsky = [] #np.zeros(shape = commonskies.shape)
    for i in range(commonskies.shape[0]):
        xsm = xrel_throughput[i] * commonskies[i]
        #print('xsm.shape: ', xsm.shape)
        #print('xrel_throughput.shape', xrel_throughput[i].shape)
        #print('fiber-to-fiber.shape: ', fiber_to_fiber[i].shape)
        xss = fiber_to_fiber[i] * xsm
        xsky.append(xss)
    xsky = np.array(xsky)
    #print('xsky.shape: ',xsky.shape)
    #print('nonnan xsky: ', xsky[np.isfinite(xsky)])
    return xsky

def scale_poly(commonskies, sky_spectra, xsky, ww ):

    """ scale the xsky spectrum with a polynomial fit to the residuals """
    res = []
    #res = np.ones(shape = commonskies.shape)
    #ww = ww # it should do to fit to np.arange(0,re.shape[0],1)...?
    for i in range(sky_spectra.shape[0]):

        re = (sky_spectra[i,45,:] - xsky[i,45,:])/ xsky[i,45,:]
        #print('re[np.isfinite] ',re[np.isfinite(re)])
        sigma = np.nanstd(re[250:])
        sigma2 = np.nanstd(re[:250])
        kappa = 3.5

        flag = np.isfinite(re[250:]) & (np.abs(re[250:]-np.nanmean(re[250:]))<kappa*sigma)
        flag2 = np.isfinite(re[:250]) & (np.abs(re[:250]-np.nanmean(re[250:]))<kappa*sigma2)

        flag2 = np.append(flag2,flag)
        #try:
        f = np.polyfit(ww[flag2], re[flag2], 5)

        g = f[5] + ww*f[4] + ww*ww*f[3] + ww*ww*ww*f[2]+ ww*ww*ww*ww*f[1]+ ww*ww*ww*ww*ww*f[0]

        xsky[i] = xsky[i] + xsky[i] * g
        re = re - g
        #print('new re nonnan: ', re[np.isfinite(re)])
        res.append(re)
        #except: # IndexError or TypeError, sometimes re is all inf or nan...
    #        print('A TypeError or IndexError occurred in polynomial fitting.')
#            pass
    res = np.array(res)
    return xsky, res

def sub_xsky(spectra, xsky):
    """ subtract (hopefully) scaled xsky """
    xskysub = spectra - xsky
    return xskysub

def sub_xsky_new(sky_subtracted, sky_spectra, xsky):
    xskysub = sky_subtracted + sky_spectra - xsky
    return xskysub

def save_fits(ff, xskysub, res, allshots, exposures, shots, overwrite):

    """ save multifits files with the new extensions 'xsky_subtracted' and 'rel_error' """

    for i in range(len(ff)):
        fin = ff[i]
        try:
            hdu = fits.open('../xsky/{}/{}/x_{}'.format(allshots[i], exposures[i], fin.split('/')[-1]))
            hdu.append(fits.ImageHDU( xskysub[i], name='xsky_subtracted'))
            hdu.append(fits.ImageHDU(xsky[i], name='xsky_spectrum_rb'))
            hdu.writeto('../xsky/{}/{}/x_{}'.format(allshots[i], exposures[i], fin.split('/')[-1]))
            print("found and wrote "+'../xsky/{}/{}/x_{}'.format(allshots[i], exposures[i], fin.split('/')[-1]))
        except:
            hdu1 = fits.open(fin)
            hdu = fits.HDUList([fits.PrimaryHDU(hdu1["spectrum"].data),hdu1["sky_spectrum"]])
            hdu1.close()
            hdu.append(fits.ImageHDU( xskysub[i], name='xsky_subtracted'))
            #hdu.append(fits.ImageHDU( res[i], name='rel_error'))
            # spectra, xsky
            hdu.append(fits.ImageHDU(spectra[i], name='spectrum_rb'))
            hdu.append(fits.ImageHDU(xsky[i], name='xsky_spectrum_rb'))
            hdu.append(fits.ImageHDU(sky_spectra[i], name='sky_spectrum_rb'))
            hdu.append(fits.ImageHDU(sky_subtracted[i], name="sky_subtracted_rb"))
            hdu.append(fits.ImageHDU(fiber_to_fiber[i], name="fiber_to_fiber_rb"))

            if overwrite:
                hdu.writeto(fin, overwrite = True)
            else:
                hdu.writeto('../xsky/{}/{}/x_{}'.format(allshots[i], exposures[i], fin.split('/')[-1]), overwrite=True)
                print("wrote "+'../xsky/{}/{}/x_{}'.format(allshots[i], exposures[i], fin.split('/')[-1]))
            hdu.close()


    hdu = fits.PrimaryHDU(np.arange(0,1,2))
    error = fits.HDUList([hdu])
    error.append(fits.ImageHDU(np.array(res)))
    error.writeto('classestesterror_{}.fits'.format(shots[0][:-4]), overwrite=True)

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



def find_indices(shots):
    # assumes that only shots from one night are used...
    night = shots[0][:-4]
    pattern = '/work/03946/hetdex/maverick/red1/reductions/{}/virus/virus0000*/exp0?/virus/multi*.fits'.format(night)

    ff = glob.glob(pattern)
    ff = np.array(ff)

    ifuslots = []
    amps = []
    exposures = []
    allshots = []

    for f in ff:
        h,t = os.path.split(f)
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

    return ff, allshots, exposures, ifuslots, amps



def main():
    REBIN_ONLY = False
    with open("shots-22.txt","r") as shotlist:
        shotlist = shotlist.read().split("\n")[:-1]
    shots = shotlist #['20180822v008','20180822v009','20180822v020','20180822v021','20180822v022','20180822v023']#['20180124v010','20180124v011']
    flagged_ifus =  {'20180822v008':['044', '093', '094', '095', '106'],
                    '20180822v009':[ '092', '094' '106'],
                    '20180822v020':['094'],
                    '20180822v021':['026', '073', '074', '094'],
                    '20180822v022':['034', '037', '072', '073', '094'],
                    '20180822v023':['052', '094'],'20180110v021':['036','042','043','074','086','095','104'],
                    '20180120v008':['036','042','043','074','086','095','104'],
                    '20180123v009':['036','042','043','074','086','095','104'],
                    '20180124v010':['036','042','043','074','086','095','104']}

    ff, allshots, exposures, ifuslots, amps = find_indices(shots)
    overwrite = False
    #print(ff)
    #print(ff.shape)
    #return
    niceifus = flag_ifus(flagged_ifus, ifuslots, allshots)
    print('flagged ifus')
    sky_spectra, spectra, fiber_to_fiber, ww = rebin(ff, allshots, exposures, ifuslots, amps)
    print('rebinned')
    if REBIN_ONLY:
        return
    sky_models = get_skymodels(sky_spectra, fiber_to_fiber)
    print('sky_models')
    commonskies = get_commonsky(sky_models, shots, allshots, exposures, niceifus)
    print('common sky')
    rel_throughput = get_rel_throughput(sky_models, commonskies)
    print('rel_throughput')
    xrel_throughput = get_xrel_throughput(rel_throughput, allshots, exposures, ifuslots, amps, niceifus)
    print('xrel_throughput')
    xsky = get_xskyspectra(commonskies, xrel_throughput, fiber_to_fiber)
    print('xsky spectra')
    xsky, res = scale_poly(commonskies, sky_spectra, xsky, ww )
    print('scaled poly')
    xskysub = sub_xsky_new(sky_subtracted, sky_spectra, xsky) #sub_xsky(spectra, xsky)
    print('xsky subtracted \(^-^)/')
    save_fits(ff, xskysub, res, allshots, exposures, shots, overwrite)
    print('saved fits')

main()
