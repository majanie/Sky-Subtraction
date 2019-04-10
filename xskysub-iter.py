from astropy.table import Table
#import astropy.units as u
#from astropy.coordinates import SkyCoord
from astropy.stats import biweight_location, biweight_scale
from hetdex_api.shot import *
from fiber_utils import bspline_x0
from scipy.interpolate import interp1d
#from scipy.signal import medfilt2d

#from hetdex_api.survey import Survey
#from scipy.interpolate import interp1d
#import os
from scipy.ndimage.filters import median_filter
#import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
import glob
from astropy.io import fits
#from astropy.stats import biweight_location
import argparse
from scipy.ndimage.filters import gaussian_filter
#from fiber_utils import bspline_x0
#from scipy.ndimage.filters import gaussian_filter

parser = argparse.ArgumentParser()
parser.add_argument("-s","--shot",type=str, default="20190101v014", help="Shot")
#parser.add_argument("-s", "--shotlist", type=str, default="jan-19.txt",
#                    help="Shotlist.")
parser.add_argument("-n","--name", type=str, default="jan-19")
parser.add_argument("--xrt",type=bool, default=False)
parser.add_argument("--gauss", type=bool, default=False)
#args = parser.parse_args(sys.argv[1:])


shot = "20181203v016"#'20190117v019'#"20190104v019"#'20181203v015'# good one
exp = "exp02"

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

# TO Do : exclude and interpolate : [3910,4359,5461]

def get_xrt_new(ifuslots):
	xrt_0, wave = pickle.load(open("xrt-2019.pickle","rb"))
	xrt = {}
	line = 3910
	here1 = np.where((wave>line-10) & (wave<line+10))[0]
	line = 4359
	here2 = np.where((wave>line-10) & (wave<line+10))[0]
	line = 5461
	here3 = np.where((wave>line-10) & (wave<line+10))[0]
	print wave
	if SMOOTHATA:
		for key in xrt_0.keys():
			here = here1
			slope = (xrt_0[key][here[-1]+1] - xrt_0[key][here[0]-1])/float(len(here))
			xrt_1 = np.concatenate([xrt_0[key][:here[0]], xrt_0[key][here[0]-1] + np.arange(len(here))*slope
                                    , xrt_0[key][here[-1]+1:]])
			here = here2
			slope = (xrt_0[key][here[-1]+1] - xrt_0[key][here[0]-1])/float(len(here))
			xrt_1 = np.concatenate([xrt_0[key][:here[0]], xrt_0[key][here[0]-1] + np.arange(len(here))*slope
                                    , xrt_0[key][here[-1]+1:]])
			here = here3
			slope = (xrt_0[key][here[-1]+1] - xrt_0[key][here[0]-1])/float(len(here))
			xrt_1 = np.concatenate([xrt_0[key][:here[0]], xrt_0[key][here[0]-1] + np.arange(len(here))*slope
                                    , xrt_0[key][here[-1]+1:]])
			#print xrt_1.shape
			xrt[key] = interp1d(wave, gaussian_filter(xrt_1, sigma=SIGMA), fill_value="extrapolate")#gaussian_filter(xrt_0[key], sigma=SIGMA), fill_value="extrapolate")
	else:
		for key in xrt_0.keys():
			xrt[key] = interp1d(wave, xrt_0[key], fill_value="extrapolate")
	return xrt

def boxes(array, flag, size):

    boxes2 = np.split(np.arange(112), size[0])
    boxes1 = np.split(np.arange(1032), size[1])
    #boxes1 = np.arange(10,258, dtype=int), np.arange(258,516, dtype=int), np.arange(516,774, dtype=int), np.arange(774, 1032, dtype=int)
    #boxes2 = np.arange(28, dtype=int), np.arange(28,56, dtype=int), np.arange(56,84, dtype=int), np.arange(84,112, dtype=int)
    newarray = np.array(array.copy(), dtype=np.float64)/np.array(array.copy())
    medfilt = np.zeros(array.shape)
    medians = np.zeros(size)#(4,4))
    for i, box1 in enumerate(boxes2):
        #print flag[box1]
        for j, box2 in enumerate(boxes1):
            medians[i,j] = np.nanmedian(array[box1][:,box2][flag[box1]])
            for k in box1:
                for l in box2:
                    medfilt[k,l] = medians[i,j]
    return gaussian_filter(medfilt, sigma=(8,100))

    # SWITCHES

SIGMA = 4
SMOOTHATA = SIGMA != 0
print SMOOTHATA

FIBER_ADJ = False
THRESHOLD = 1.03 # last time 1.05, too high
LOWER = 0.96
ADJUSTMENT = THRESHOLD == 0
FILTERSIZE = (10,200)
FILTER = True
KAPPA = 3.

SKY_FROM_SPEC = True

def_wave = np.arange(3470., 5542., 2.)

fileh = open_shot_file(shot) # args.shot in .py

table =  Table(fileh.root.Data.Fibers.read())
print table.columns

expnum = table["expnum"].data

fix = np.array(table["fibidx"].data)#[expnum==1]
fix = np.array(np.split(fix, fix.shape[0]/112))

if SKY_FROM_SPEC:
    sky_spectra_orig = table["spectrum"].data #- table["sky_subtracted"].data
else:
    sky_spectra_orig = table["spectrum"].data - table["sky_subtracted"].data
sky_spectra_orig = np.array(sky_spectra_orig, dtype=np.float64)#[expnum==1]
#sky_spectra = np.array(np.split(sky_spectra, sky_spectra.shape[0]/112))

sky_subtracted_orig = table["sky_subtracted"].data
sky_subtracted_orig = np.array(sky_subtracted_orig, dtype=np.float64)#[expnum==1]
#sky_subtracted = np.array(np.split(sky_subtracted, sky_subtracted.shape[0]/112))

fiber_to_fiber_orig = table["fiber_to_fiber"].data
fiber_to_fiber_orig = np.array(fiber_to_fiber_orig, dtype=np.float64)#[expnum==1]
#fiber_to_fiber = np.array(np.split(fiber_to_fiber, fiber_to_fiber.shape[0]/112))

wavelength_orig = table["wavelength"].data
wavelength_orig = np.array(wavelength_orig, dtype=np.float64)#[expnum==1]
#wavelength = np.array(np.split(wavelength, wavelength.shape[0]/112))

ifuslots = table["ifuslot"].data
ifuslots = np.array([x[0] for x in np.split(ifuslots, ifuslots.shape[0]/112)])

amps = table["amp"].data
amps = np.array([x[0] for x in np.split(amps, amps.shape[0]/112)])

exposures = np.array(["exp0{}".format(x) for x in table["expnum"].data])
exposures = np.array([x[0] for x in np.split(exposures, exposures.shape[0]/112)])

MYATA = True

if MYATA:
    xrt = get_xrt_new(ifuslots)
else:
    wl = np.arange(3480, 5551, 2)
    wl.shape

    ata = table['Amp2Amp'].data
    xrt = {}
    for i in range(len(ifuslots)):
        b = ata[(table['ifuslot']==ifuslots[i])&(table['amp']==amps[i])&(table['expnum']==2)][0]
        if len(b[b!=0]) ==0:
            print 'bad amp '+ifuslots[i]+amps[i]
            continue
        xrt[(ifuslots[i],amps[i])] = interp1d(wl[b!=0], b[b!=0], fill_value='extrapolate')

rescor = get_rescor(ifuslots, amps, def_wave)
#midftf = pickle.load(open("midftf.pkl","rb"))
#xrt = get_xrt_new(ifuslots)

meanmean = np.nanmedian(sky_spectra_orig[:,600:900], axis=1)
#plt.figure()
#plt.hist(meanmean, bins=np.arange(0,500,10))
biw = biweight_location(meanmean)
biscale = biweight_scale(meanmean)
#plt.axvline(biw, color='k')
#plt.axvline(biw+KAPPA*biscale, color='k')

if False:
    flag = meanmean < biw + KAPPA*biscale
else:
    flag = meanmean == meanmean

sky_spectra = np.array(np.split(sky_spectra_orig, sky_spectra_orig.shape[0]/112))[exposures==exp]
sky_subtracted = np.array(np.split(sky_subtracted_orig, sky_subtracted_orig.shape[0]/112))[exposures==exp]
fiber_to_fiber = np.array(np.split(fiber_to_fiber_orig, fiber_to_fiber_orig.shape[0]/112))[exposures==exp]
wavelength = np.array(np.split(wavelength_orig, wavelength_orig.shape[0]/112))[exposures==exp]

ifuslots_exp = ifuslots[exposures==exp]
amps_exp = amps[exposures==exp]

flag = np.array(np.split(flag, flag.shape[0]/112))[exposures==exp]

sky_spectra_iter = sky_spectra.copy()

for i in range(sky_spectra_iter.shape[0]):
    try:
        sky_spectra_iter[i] = sky_spectra_iter[i]/(xrt[(ifuslots[i], amps[i])](wavelength[i])*fiber_to_fiber[i])
    except KeyError:
        sky_spectra_iter[i] *= 0

binsize = int(sky_spectra_iter[flag].size/3000.)
nwave_iter, smooth_iter = make_avg_spec(wavelength[flag], sky_spectra_iter[flag], binsize=binsize)
#nwave, smooth = get_common_sky(sky_spectra, fiber_to_fiber, wavelength)

csm_iter = interp1d(nwave_iter, smooth_iter, fill_value="extrapolate")

# get new (iterated) sky as csm * xrt * f2t * factor
tmp=[]
new_sky_iter = sky_spectra_iter.copy()
for i in range(new_sky_iter.shape[0]):
    try:
        factor = np.nanmedian(new_sky_iter[i]/csm_iter(wavelength[i]))#np.nanmedian(new_sky_iter[i]/csm_iter(wavelength[i]))
        tmp.append(factor)
        if factor > THRESHOLD:
            factor = THRESHOLD
        if factor < LOWER:
            factor = LOWER
        new_sky_iter[i] = csm_iter(wavelength[i]) * xrt[(ifuslots[i], amps[i])](wavelength[i]) * factor * fiber_to_fiber[i]
    except KeyError:
        new_sky_iter[i] *= 0

xskysub_iter = []
medfilt_size = FILTERSIZE  #(50,50)#(70,200)
medianfilters = []
N = len(sky_spectra)

START, STOP = 0,N

for i in range(len(sky_spectra))[START:STOP]:
	if SKY_FROM_SPEC:
		new_skysub = sky_spectra[i] - new_sky_iter[i]*(1+rescor[(ifuslots[i], amps[i])])
	else:
		new_skysub = sky_subtracted[i] + sky_spectra[i]*(1+rescor[(ifuslots[i], amps[i])]) - xsky_spectra[i]*(1+rescor[(ifuslots[i], amps[i])])
	if FILTER:
		medfilt = boxes(new_skysub, np.ones((new_skysub.shape[0],), dtype=bool), size=(4,4))
		medianfilters.append(medfilt)
		new_skysub -= medfilt
	new_skysub_int = []
	new_skysub[sky_spectra[i]==0] = 0
	for j in range(112):
 		dw = np.diff(wavelength[i,j])
 		dw = np.hstack((dw[0], dw))
		#dw = 1
		new_skysub_int.append(np.interp(def_wave, wavelength[i,j], new_skysub[j]/dw, left=0.0, right=0.0))
	xskysub_iter.append(new_skysub_int)

	print( "done {} / {} ".format(i, N))
xskysub_iter = np.array(xskysub_iter)

#############################################################
# ---------------------Now iterating------------------------#
#############################################################

meanmean = np.nanmedian(first_sky_subtracted[:,600:900], axis=1)
#plt.figure()
#plt.hist(meanmean, bins=np.arange(-5,30,1))
biw = biweight_location(meanmean)
biscale = biweight_scale(meanmean)
#plt.axvline(biw, color='k')
#plt.axvline(biw+KAPPA*biscale, color='k')

if True:
    flag = meanmean < biw + KAPPA*biscale
else:
    flag = meanmean == meanmean

sky_spectra = np.array(np.split(sky_spectra_orig, sky_spectra_orig.shape[0]/112))[exposures==exp]
sky_subtracted = np.array(np.split(sky_subtracted_orig, sky_subtracted_orig.shape[0]/112))[exposures==exp]
fiber_to_fiber = np.array(np.split(fiber_to_fiber_orig, fiber_to_fiber_orig.shape[0]/112))[exposures==exp]
wavelength = np.array(np.split(wavelength_orig, wavelength_orig.shape[0]/112))[exposures==exp]

flag = np.array(np.split(flag, flag.shape[0]/112))#[exposures==exp]

sky_spectra_iter = sky_spectra.copy()
sky_spectra_iter.shape

for i in range(sky_spectra_iter.shape[0]):
    try:
        sky_spectra_iter[i] = sky_spectra_iter[i]/(xrt[(ifuslots[i], amps[i])](wavelength[i])*fiber_to_fiber[i]*tmp[i])
    except KeyError:
        sky_spectra_iter[i] *= 0

binsize = int(sky_spectra_iter[flag].size/3000.)
nwave_iter, smooth_iter = make_avg_spec(wavelength[flag], sky_spectra_iter[flag], binsize=binsize)
#nwave, smooth = get_common_sky(sky_spectra, fiber_to_fiber, wavelength)

csm_iter = interp1d(nwave_iter, smooth_iter, fill_value="extrapolate")

THRESHOLD = 1.0031
# get new (iterated) sky as csm * xrt * f2t * factor
tmp2=[]
new_sky_iter = sky_spectra_iter.copy()
for i in range(new_sky_iter.shape[0]):
    try:
        if flag[i][flag[i]].size > 30:
            factor = np.nanmedian(new_sky_iter[i][flag[i]]/csm_iter(wavelength[i][flag[i]]))#np.nanmedian(new_sky_iter[i]/csm_iter(wavelength[i]))
            tmp2.append(factor)
        else:
            factor = 1#np.nanmedian(new_sky_iter[i][flag[i]]/csm_iter(wavelength[i][flag[i]]))#np.nanmedian(new_sky_iter[i]/csm_iter(wavelength[i]))
            tmp2.append(factor)
        if factor > THRESHOLD:
            factor = THRESHOLD
        new_sky_iter[i] = csm_iter(wavelength[i]) * xrt[(ifuslots[i], amps[i])](wavelength[i]) * factor * fiber_to_fiber[i]
    except KeyError:
        new_sky_iter[i] *= 0

second_skysub_iter = []
medfilt_size = (80,100)#FILTERSIZE  #(50,50)#(70,200)
medianfilters = []
N = len(sky_spectra)

FILTER = True

FILTERSIZE = (9, 99)

START, STOP = 0,N

size = (8,8)

for i in range(len(sky_spectra))[START:STOP]:
	if SKY_FROM_SPEC:
		new_skysub = sky_spectra[i] - new_sky_iter[i]*(1+rescor[(ifuslots[i], amps[i])])
	else:
		new_skysub = sky_subtracted[i] + sky_spectra[i]*(1+rescor[(ifuslots[i], amps[i])]) - xsky_spectra[i]*(1+rescor[(ifuslots[i], amps[i])])
	if FILTER:
		medfilt = boxes(new_skysub, flag[i], size=size)#newboxcar(new_skysub, flag[i], FILTERSIZE) #median_filter(new_skysub[flag[i]], size=medfilt_size)
		medfilt[~np.isfinite(medfilt)] = 0
		medianfilters.append(medfilt)
		new_skysub -= medfilt
	new_skysub_int = []
	new_skysub[sky_spectra[i]==0] = 0
	for j in range(112):
 		dw = np.diff(wavelength[i,j])
 		dw = np.hstack((dw[0], dw))
		#dw = 1
		new_skysub_int.append(np.interp(def_wave, wavelength[i,j], new_skysub[j]/dw, left=0.0, right=0.0))
	second_skysub_iter.append(new_skysub_int)

	print( "done {} / {} ".format(i, N))
second_skysub_iter = np.array(second_skysub_iter)

tickarray = np.array(np.split((np.arange(flag.size)), flag.shape[0]))[START:STOP][np.where(~flag[START:STOP])]

ampticks = np.arange(N)*112

plt.figure(figsize=(25,200))
plt.subplot(131)
plt.title("original")
plt.imshow(np.concatenate(sky_subtracted[START:STOP]), vmin=-40, vmax=40, interpolation="none", aspect="auto", cmap="Greys_r")
plt.yticks(tickarray - START*112, 'x'*len(tickarray), color='red');
#plt.colorbar();
plt.subplot(132)
plt.title("xskysub")
plt.imshow(np.concatenate(second_skysub_iter), vmin=-20, vmax=20, interpolation="none", aspect="auto", cmap="Greys_r")
plt.yticks(ampticks, [ifuslots[i]+amps[i] for i in range(N)], rotation=90)#(tickarray - START*112, 'x'*len(tickarray), color='red')
#plt.colorbar();
plt.subplot(133)
plt.title("median filter")
plt.imshow(np.concatenate(medianfilters), vmin=-20, vmax=40, interpolation="none", aspect="auto", cmap="Greys_r");
plt.yticks(ampticks, [ifuslots[i]+amps[i] for i in range(N)], rotation=90);
#plt.colorbar(orientation="horizontal");
plt.savefig('fullframe-{}-newamps.png'.format(shot), bbox_inches='tight')
