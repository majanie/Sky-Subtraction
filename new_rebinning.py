from astropy.io import fits
import numpy as np
import os
import glob

# changed to raise error

# this function will create a linearely binned version
# of a particular spectrum
from srebin import linlin

# start = 3494.74, step =  1.9858398, stop = 5500.
# changed: removed extensions spectrum and sky_subtracted

def get_rebinned(ff,exposures, ifuslots, amps, exp, ifuslot, amp, extensions=['spectrum', 'sky_spectrum', 'fiber_to_fiber'], start = 3494.74, step =  1.9858398, stop = 5500.):
    
    #global exposures, ifuslots, amps#, ff
 
    ii = (exposures == exp) * (ifuslots == ifuslot) * (amps == amp)
    if not sum(ii) == 1:
        print('ValueError in Rebinning')
        raise ValueError("Error: Could not identify file for exposure {} slot {} and amp {}.".format(exp, ifuslot, amp))
        return
  
    fin = ff[ii][0]
    print("Reading {}".format(fin))
    hdu = fits.open(fin)
  
    wl = hdu['wavelength'].data
    #spectrum       = hdu['spectrum'].data
    #sky_spectrum   = hdu['sky_spectrum'].data
    #sky_subtracted = hdu['sky_subtracted'].data

    #start,stop = 3503.9716796, 5396.477
    N = int( np.ceil( (stop - start)/step ) )
    
    rebinned = {}
    for ext in extensions:
        #for j in range(hdu[ext].data.shape[1]): # This might cause big errors...
            #isnull = np.unique(hdu[ext].data[:,j])
            #isnull = isnull[np.where(isnull != 0)]
            #if len(isnull)==0:
            #    hdu[ext].data[:, j] = np.ones(hdu[ext].data.shape[0])*np.nan
        print("Rebinning {}".format(ext))
        
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

def main(pattern, night, shot, extensions = ['spectrum','sky_spectrum', 'fiber_to_fiber']):
    
    rebinned = {}

    #night = "20180601"
    #shot = "013"
    #pattern = 'work/03946/hetdex/maverick/red1/reductions/{}/virus/virus0000{}/exp0?/virus/multi*.fits'.format(night, shot)

    ff = glob.glob(pattern)
    ff = np.array(ff)

    # extract ifuslot, amplifier and exposure from file name
    ifuslots = []
    amps = []
    exposures = []

    for f in ff:
        h,t = os.path.split(f)
        ifuslots.append( t[10:13] )
        amps.append( t[18:20] )
        exposures.append( h.split('/')[-2])

    ifuslots = np.array(ifuslots)
    amps = np.array(amps)
    exposures = np.array(exposures)

    #ee, ss, aa = np.sort(np.unique(exposures)), np.sort(np.unique(ifuslots)), np.sort(np.unique(amps))
    ee, ss, aa = np.unique(exposures), np.unique(ifuslots), np.unique(amps)
    #fin = ff[0]
    #print("Reading {}".format(fin))
    #hdu = fits.open(fin)
    #ww = []
    print(ee, ss, aa)
    for exp in ee[:]:
        for ifuslot in ss[:]:
            for amp in aa[:]:
                try:
                    ww, rebinned[(exp, ifuslot, amp)] = get_rebinned(ff,exposures, ifuslots, amps, exp, ifuslot, amp, extensions=extensions) # extensions instead of ext
                except ValueError:
                    pass

    return ww, rebinned
