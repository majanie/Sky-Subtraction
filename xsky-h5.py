import pickle
import tables as tb
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import sys
from astropy.table import Table
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.stats import biweight_location
from hetdex_api.shot import *
from fiber_utils import bspline_x0
from scipy.interpolate import interp1d
from scipy.signal import medfilt2d

from hetdex_api.survey import Survey
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--shot", type=str, default="20190101v011",
                    help="Shotid.")
args = parser.parse_args(sys.argv[1:])

def make_avg_spec(wave, spec, binsize=35, knots=None):
    if knots is None:
        knots = wave.shape[1]
    ind = np.argsort(wave.ravel())
    N, D = wave.shape
    wchunks = np.array_split(wave.ravel()[ind],
                             N * D / binsize)
    schunks = np.array_split(spec.ravel()[ind],
                             N * D / binsize)
    nwave = np.array([np.mean(chunk) for chunk in wchunks])
    B, c = bspline_x0(nwave, nknots=knots)
    nspec = np.array([biweight_location(chunk) for chunk in schunks])
    sol = np.linalg.lstsq(c, nspec)[0]
    smooth = np.dot(c, sol)
    return nwave, smooth

def get_xrt():
        xrt_0 = pickle.load(open("ratios-mid.pkl","rb"))
        xrt = {}
        for key in xrt_0.keys():
                #wave = np.arange(3510,5490,20)
                #a = xrt_0[key]#[50]
                #tmp = []
                #for i, ww in enumerate(wave):
        #               if i == 0:
        #                       tmp.append(np.nanmedian(a[(def_wave<=ww+10)&(a!=0)]))
        #               elif i == len(wave)-1:
        #                       tmp.append(np.nanmedian(a[(def_wave>=ww-10)&(a!=0)]))
        #               else:
        #                       tmp.append(np.nanmedian(a[(def_wave>=ww-10)&(def_wave<=ww+10)]))
                #tmp = interp1d(def_wave, xrt_0[key][50], fill_value="extrapolate")
                xrt[key] = interp1d(def_wave, xrt_0[key], fill_value="extrapolate")
        return xrt

def get_rescor(ifuslots, amps, def_wave):
    rescor_pattern = "/work/03946/hetdex/maverick/virus_config/rescor/{}res.fits" # insert multi-name without the fits (multi_???_ifu_???_am)
    ifus = np.unique(ifuslots)
    aa = np.unique(amps)
    rescor = {}
    for ifu in ifus:
        for amp in aa:
            try:
                gg = glob.glob(rescor_pattern.format("multi_???_{}_???_{}".format(ifu, amp)))[0]
                rc = fits.open(gg)[0].data
                #fin = ff[np.where((ifuslots==ifu)*(amps==amp))][0]
                #wl = fits.open(fin)["wavelength"].data
                #rc_1 = []
                #for i in range(112):
                #    rc_1.append(np.interp(def_wave, wl[i], rc[i], left=0.0, right=0.0))
                rescor[(ifu, amp)] = np.array(rc)
            except IndexError:
                #print("rescor not found for ifuslot {} amp {} ".format(ifu, amp))
                rescor[(ifu, amp)] = np.zeros((112,1032))
                pass
    return rescor

def get_xsky(csm, sky_spectra, fiber_to_fiber, wavelength, xrt):
        xsky = []
        facfile = {}
	print(sky_spectra.shape[0])
        for i in range(sky_spectra.shape[0]):
                ifu = ifuslots[i]
                amp = amps[i]
                xsky_1 = []
                factors = []
                for j in range(sky_spectra.shape[1]): # looping through fibers
                        wl_0 = wavelength[i,j]
                        #if args.gauss:
                        #        xrt_0 = gaussian_filter(xrt[(ifu,amp)](wl_0), sigma=1)
                        #else:
                        xrt_0 = xrt[(ifu,amp)](wl_0)
                        csm_0 = csm(wl_0)
                        factors.append(np.nanmedian(sky_spectra[i,j]/(fiber_to_fiber[i,j] * xrt_0 * midftf[(ifu,amp)][::-1][j]*csm_0)))
                        xsky_0 = csm(wl_0) * xrt_0 * fiber_to_fiber[i,j] * midftf[(ifu,amp)][::-1][j] #* factor
                        xsky_1.append(xsky_0)
                midfac = np.nanmedian(factors)
                if midfac >= 1.1:
                        midfac = 1.1
                xsky.append(np.array(xsky_1)*midfac)
                #facfile[(ifuslots[i], amps[i])] = np.nanmedian(factors)
        xsky = np.array(xsky)
        #facfile_name = open("facfile-spec-{}.pkl".format(args.shot),"wb")
        #pickle.dump(facfile, facfile_name, protocol=2)
	#print "dumped facfile-spec-{}.pkl".format(args.shot)
        return xsky

def get_ratio(csm, sky_spectra, fiber_to_fiber, wavelength, xrt):
        xsky = []
        ratio = {}
        for i in range(sky_spectra.shape[0]):
                ifu = ifuslots[i]
                amp = amps[i]
                xsky_1 = []
                ratio_0 = []
                for j in range(sky_spectra.shape[1]):
                        wl_0 = wavelength[i,j]
                        xrt_0 = xrt[(ifu,amp)](wl_0)
                        csm_0 = csm(wl_0)
                        ratio_0.append(interp1d(wl_0,sky_spectra[i,j]/(fiber_to_fiber[i,j]*csm_0), fill_value="extrapolate"))
                        #factor = np.nanmedian(sky_spectra[i,j]/(fiber_to_fiber[i,j]*xrt_0*csm_0))
                        #xsky_0 = csm(wl_0) * xrt_0 * fiber_to_fiber[i,j] * factor
                        #xsky_1.append(xsky_0)
                #xsky.append(xsky_1)
                ratio[(ifuslots[i],amps[i])] = np.nanmedian([x(def_wave) for x in ratio_0],axis=0)
	ratio_file = open("ratio-spec-{}.pkl".format(args.shot),"wb")
	pickle.dump(ratio, ratio_file, protocol=2)
        #xsky = np.array(xsky)
	print "dumped ratio-spec-{}.pkl".format(args.shot)
        return ratio



def_wave = np.arange(3470., 5542., 2.)

#survey = Survey('hdr1')

fileh = open_shot_file(args.shot)

#fibers = Fibers(args.shot)

table =  Table(fileh.root.Data.Fibers.read())

#<TableColumns names=('multiframe','ra','dec','Amp2Amp','Throughput','amp','calfib','calfibe','contid','error1Dfib','expnum','fiber_to_fiber','fibnum','fpx','fpy','ifuid','ifuslot','ifux','ifuy','obsind','sky_subtracted','specid','spectrum','trace','twi_spectrum','wavelength')>
here = table["expnum"]==1
spectra = table["spectrum"][here] - table["sky_subtracted"][here]
print spectra.shape

fiber_to_fiber = table["fiber_to_fiber"][here]
print fiber_to_fiber.shape

spectra_1 = spectra / fiber_to_fiber
print spectra_1.shape

midmean = np.nanmean(spectra_1[:,500:750], axis=1)
"""
plt.figure()
plt.hist(midmean, bins = np.arange(50,300,5))
plt.axvline(130)
plt.axvline(170)
plt.title("midmean")
plt.show()

midmax = np.nanmax(spectra_1[:,:], axis=1)
plt.figure()
plt.hist(midmax, bins = np.arange(200,1000,10))
plt.axvline(250)
plt.axvline(350)
plt.title("midmax")
plt.show()
"""
"""midmin = np.nanmin(spectra_1[:,:], axis=1)
plt.figure()
plt.hist(midmin, bins = np.arange(0,50,1))
plt.axvline(1)
#plt.axvline(200)
plt.title("midmin")
plt.show()"""

flag = [True for i in range(len(midmean))]
#flag = (midmean>=130)*(midmean<=170)*(midmax>=250)*(midmax<=350)
#print "percentage remaining : ", len(flag[flag])/float(len(flag))

wavelength = table["wavelength"].data[here]
ifuslots = table["ifuslot"].data[here]
amps = table["amp"].data[here]

print ifuslots.shape
print amps.shape

#print spectra[0]
binsize=int(spectra_1[flag].size/3000.)
print binsize
nwave, smooth = make_avg_spec(wavelength[flag], spectra_1.data[flag], binsize=binsize)
csm = interp1d(nwave, smooth, fill_value="extrapolate")

xrt = get_xrt()
midftf = pickle.load(open("midftf.pkl","rb"))
#tmp =  get_ratio(csm, np.array(np.array_split(spectra,spectra.shape[0]/112)), np.array(np.array_split(fiber_to_fiber,fiber_to_fiber.shape[0]/112)),
#         np.array(np.array_split(wavelength,wavelength.shape[0]/112)), xrt)


xsky = get_xsky(csm, np.array(np.array_split(spectra,spectra.shape[0]/112)), np.array(np.array_split(fiber_to_fiber,fiber_to_fiber.shape[0]/112)),
	 np.array(np.array_split(wavelength,wavelength.shape[0]/112)), xrt)

rescor = get_rescor(ifuslots, amps, def_wave)

xskysub = []
medfilt_size = (51,51)
N = len(xsky)
spectra = np.array(np.array_split(spectra, spectra.shape[0]/112))
wavelength = np.array(np.array_split(wavelength,wavelength.shape[0]/112))
for i in range(len(xsky))[:10]:

        new_skysub = spectra[i] - xsky[i]*(1+rescor[(ifuslots[i], amps[i])])
	#sky_subtracted[i] + sky_spectra[i]*(1+rescor[(ifuslots[i], amps[i])]) - xsky_spectra[i]*(1+rescor[(ifuslots[i], amps[i])])
        medfilt = medfilt2d(new_skysub, kernel_size=medfilt_size)
        new_skysub -= medfilt
        new_skysub_int = []
        for j in range(112):
                new_skysub_int.append(np.interp(def_wave, wavelength[i,j], new_skysub[j], left=0.0, right=0.0))
        xskysub.append(new_skysub_int)

        print( "done {} / {} ".format(i, N))

xskysub = np.array(xskysub)

xskysub = np.concatenate(xskysub)

print xsky.shape
print(np.concatenate(xsky).shape)

plt.figure()
##plt.imshow(spectra - np.concatenate(xsky), vmin=-20, vmax=20, aspect="auto",interpolation="none")
plt.imshow(xskysub, vmin=-20, vmax=20, aspect="auto",interpolation="none")
plt.show()
#plt.matshow(table["sky_subtracted"].data, vmin=-20, vmax=20)
#plt.show()
