import sys
import os
import glob
import numpy as np
from astropy.io import fits

# this function will create a linearely binned version
# of a particular spectrum
from srebin import linlin

def print_no_newline(string):
    sys.stdout.write(string)
    sys.stdout.flush()
# allcubes should include all shots etc in one night.

class allcubes:
    def __init__(self, shots, ff, allshots, exposures, ifuslots, amps, overwrite, flagged_ifus):
        self.shots = shots
        self.ff = ff
        self.allshots = allshots
        self.exposures = exposures
        self.ifuslots = ifuslots
        self.amps = amps
        self.overwrite = overwrite
        self.flagged_ifus = flagged_ifus
        self.sky_spectra = []
        self.spectra = []
        self.fiber_to_fiber = []
        self.ww = []
        self.sky_models = []
        self.commonskies = []
        self.rel_throughtput = []
        self.xrel_throughput = []
        self.xsky = []
        self.xskysub = []
        self.res = []
        
    def flag_ifus(self):
        
        """ self.niceifus is a boolean array that indicates which ifus are not flagged (without stars) """
        
        self.niceifus = np.ones(shape=self.ifuslots.shape, dtype=np.int8)*True
        for key in self.flagged_ifus.keys():
            for flagged_ifu in self.flagged_ifus[key]:
                self.niceifus[(self.ifuslots==flagged_ifu)*(self.allshots == key)] = False
        #print(np.unique(self.ifuslots[np.where(self.niceifus)]))
        
    def rebin(self):
        
        """ turnes sky spectrum, spectrum, and fiber-to-fiber arrays into linearly binned versions """
        number = self.ff.shape[0]
        i = 0 
        for fin in self.ff:
            i += 1
            ww, rebinned = get_rebinned(fin)
            print('{}/{}'.format(i, number))
            self.sky_spectra.append(rebinned['sky_spectrum'])
            self.spectra.append(rebinned['spectrum'])
            self.fiber_to_fiber.append(rebinned['fiber_to_fiber'])
        self.ww = ww 
        self.sky_spectra = np.array(self.sky_spectra)
        self.spectra = np.array(self.spectra)
        self.fiber_to_fiber = np.array(self.fiber_to_fiber)
            
    def get_skymodels(self):
        
        """ get (mean) sky for each amplifier """
        
        for i in range(self.sky_spectra.shape[0]):
            sky_model = np.nanmean( self.sky_spectra[i] / self.fiber_to_fiber[i], axis = 0)
            self.sky_models.append(sky_model)
        self.sky_models = np.array(self.sky_models)
        print('sky_models shape: ', self.sky_models.shape)
        
    def get_commonsky(self):
        
        """ get mean sky for each shot and exposure """
        
        self.commonskies = np.zeros(shape = self.sky_models.shape)
        for shot in self.shots:
            for exp in ['exp01','exp02','exp03']:
                thisnightandexp = np.where((self.allshots==shot)&(self.exposures==exp)&(self.niceifus))
                commonsky = np.nanmean( self.sky_models[thisnightandexp] , axis = 0)
                self.commonskies[thisnightandexp] = commonsky
                
    def get_rel_throughput(self):
        
        """ get relative throughput of each amplifier """
        
        self.rel_throughput = np.zeros(shape = self.sky_models.shape)
        for i in range(self.sky_models.shape[0]):
            self.rel_throughput[i] = self.sky_models[i] / self.commonskies[i]
        
    def get_xrel_throughput(self):
        
        """ get mean relative throughput without using flagged ifus and the shot itself """
        
        self.xrel_throughput = np.zeros(shape = self.rel_throughput.shape)
        for i in range(self.rel_throughput.shape[0]):
            shot, exp, ifu, amp = self.allshots[i], self.exposures[i], self.ifuslots[i], self.amps[i]
            here = (self.allshots!=shot)&(self.exposures==exp)&(self.ifuslots==ifu)&(self.amps==amp)&(self.niceifus) 
            
            if sum(here) == 0:
                print('No xrt could be computed for shot {} exp {} ifu {} amp {}.'.format(shot, exp, ifu, amp))
                self.xrel_throughput[i] = self.rel_throughput[i]
            else:
                xrelthrough = np.nanmean(self.rel_throughput[here], axis = 0)
                self.xrel_throughput[i] = xrelthrough
       
    def get_xskyspectra(self):
        
        """ the xsky spectrum is the common sky times mean rel throughput times fiber-to-fiber array
            and has the same shape as the rebinned sky spectrum """
        
        self.xsky = [] #np.zeros(shape = self.commonskies.shape)
        for i in range(self.commonskies.shape[0]):
            xsm = self.xrel_throughput[i] * self.commonskies[i]
            print('xsm.shape: ', xsm.shape)
            print('xrel_throughput.shape', self.xrel_throughput[i].shape)
            print('fiber-to-fiber.shape: ', self.fiber_to_fiber[i].shape)
            xss = self.fiber_to_fiber[i] * xsm
            self.xsky.append(xss)
        self.xsky = np.array(self.xsky)
        print('xsky.shape: ',self.xsky.shape)
        print('nonnan xsky: ', self.xsky[np.isfinite(self.xsky)])
            
    def scale_poly(self):
        
        """ scale the xsky spectrum with a polynomial fit to the residuals """
        
        self.res = np.zeros(shape = self.commonskies.shape)
        ww = self.ww # it should do to fit to np.arange(0,re.shape[0],1)...?
        for i in range(self.sky_spectra.shape[0]):
           
            re = (self.sky_spectra[i,45,:] - self.xsky[i,45,:])/ self.xsky[i,45,:]
            print('re[np.isfinite] ',re[np.isfinite(re)])
            sigma = np.nanstd(re[250:])
            sigma2 = np.nanstd(re[:250])
            kappa = 3.5

            flag = np.isfinite(re[250:]) & (np.abs(re[250:]-np.nanmean(re[250:]))<kappa*sigma)
            flag2 = np.isfinite(re[:250]) & (np.abs(re[:250]-np.nanmean(re[250:]))<kappa*sigma2)

            flag2 = np.append(flag2,flag)
            try:
                f = np.polyfit(ww[flag2], re[flag2], 5)

                g = f[5] + ww*f[4] + ww*ww*f[3] + ww*ww*ww*f[2]+ ww*ww*ww*ww*f[1]+ ww*ww*ww*ww*ww*f[0]
                                     
                self.xsky[i] = self.xsky[i] + self.xsky[i] * g
                re = re - g #/ self.xsky[i,45,:]
                print('new re nonnan: ', re[np.isfinite(re)])
                self.res[i] = re
            except: # IndexError or TypeError
                print('A TypeError or IndexError occurred in polynomial fitting.')
                pass       
          
    def sub_xsky(self):
        """ subtract (hopefully) scaled xsky """
        self.xskysub = self.spectra - self.xsky
        
    def save_fits(self):
        
        """ save multifits files with the new extensions 'xsky_subtracted' and 'rel_error' """
        
        for i in range(len(self.ff)):
            fin = self.ff[i]
            hdu = fits.open(fin)
            hdu.append(fits.ImageHDU( self.xskysub[i], name='xsky_subtracted'))
            hdu.append(fits.ImageHDU( self.res[i], name='rel_error'))

            if self.overwrite:
                hdu.writeto(fin, overwrite = True)
            else: 
                hdu.writeto('classes{}_{}_{}'.format(self.allshots[i], self.exposures[i], fin.split('/')[-1]), overwrite=True)
            hdu.close()
            

        hdu = fits.PrimaryHDU(np.arange(0,1,2))
        error = fits.HDUList([hdu])
        error.append(fits.ImageHDU(np.array(self.res)))
        error.writeto('classestesterror_{}.fits'.format(self.shots[0][:-4]), overwrite=True)

            
    def process(self):
        self.flag_ifus()
        print('flagged ifus')
        self.rebin()
        print('rebinned')
        self.get_skymodels()
        print('sky_models')
        self.get_commonsky()
        print('common sky')
        self.get_rel_throughput()
        print('rel_throughput')
        self.get_xrel_throughput()
        print('xrel_throughput')
        self.get_xskyspectra()
        print('xsky spectra')
        self.scale_poly()
        print('scaled poly')
        self.sub_xsky()
        print('xsky subtracted \(^-^)/')
        self.save_fits()
        print('saved fits')
        
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
    shots = ['20180124v010','20180124v011']
    flagged_ifus =  {'20180110v021':['036','042','043','074','086','095','104'],
                    '20180120v008':['036','042','043','074','086','095','104'],
                    '20180123v009':['036','042','043','074','086','095','104'],
                    '20180124v010':['036','042','043','074','086','095','104']}
    
    ff, allshots, exposures, ifuslots, amps = find_indices(shots)
        
    cubes = allcubes(shots, ff,  allshots, exposures, ifuslots, amps, overwrite=False, flagged_ifus=flagged_ifus)
    cubes.process()
        
main()
        
        
