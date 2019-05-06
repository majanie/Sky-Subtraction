from scipy.optimize import curve_fit
from srebin import linlin
import matplotlib
matplotlib.use('agg')
import matplotlib.pylab as pylab
params = {'legend.fontsize':'x-large',
	'figure.figsize':(15,5),
	'axes.labelsize':'x-large',
	'axes.titlesize':'x-large',
	'xtick.labelsize':'x-large',
	'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)
from astropy.table import Table
from astropy.stats import biweight_location, biweight_scale
from astropy.io import fits, ascii
from scipy.ndimage.filters import gaussian_filter
from fiber_utils import bspline_x0
from scipy.interpolate import interp1d
import tables as tb
import numpy as np
import pickle
import matplotlib.pyplot as plt
import glob
import argparse
import sys

# get shot and exposure from stdin
parser = argparse.ArgumentParser()
parser.add_argument('-s','--shot',type=str,default='20190101v014',help='Shot')
parser.add_argument('-e','--exp',type=str,default='exp02',help='Exposure')
args = parser.parse_args(sys.argv[1:])

shot, exp = args.shot, args.exp

#def gaus(x,a,x0,sigma):
#	return a*np.exp(-(x-x0)**2/(2*sigma**2))

def make_avg_spec(wave, spec, binsize=35, knots=None):
	if knots is None:
		knots = wave.shape[1]
	ind = np.argsort(wave.ravel())
	N, D = wave.shape
	wchunks = np.array_split(wave.ravel()[ind],
				N*D / binsize)
	schunks = np.array_split(spec.ravel()[ind],
				N*D / binsize)
	nwave = np.array([np.mean(chunk) for chunk in wchunks])
	B,c = bspline_x0(nwave, nknots=knots)
	nspec = np.array([biweight_location(chunk) for chunk in schunks])
	sol = np.linalg.lstsq(c, nspec)[0]
	smooth = np.dot(c,sol)
	return nwave, smooth

def get_rebinned(j , inarray, start = 3494.74, step =  1.9858398, stop = 5500.):


	wl = wavelength[j] #hdu['wavelength'].data

	#start,stop = 3503.9716796, 5396.477
	N = int( np.ceil( (stop - start)/step ) )
	new = np.zeros([wl.shape[0], N])
	hduextdata = inarray
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

	return lw, new

def get_rescor(ifuslots, amps, def_wave):
	rescor_pattern = "/work/03946/hetdex/maverick/virus_config/rescor/{}res.fits"
	ifus = np.unique(ifuslots)
	aa = np.unique(amps)
	rescor = {}
	for ifu in ifus:
		for amp in aa:
			try:
				gg = glob.glob(rescor_pattern.format("multi_???_{}_???_{}".format(ifu, amp)))[0]
				rc = fits.open(gg)[0].data
				rescor[(ifu, amp)] = np.array(rc)
			except IndexError:
				rescor[(ifu, amp)] = np.zeros((112,1032))
				pass
	return rescor

def get_xrt_new():
	xrt_0, wave = pickle.load(open('xrt-2019.pickle','rb'))
	xrt = {}
	weirdampslist = [[('035','LL'),590, 615],[('082','RL'),654,681],[('023','RL'), 349, 376],[('026','LL'), 95,142]]
	for amp in weirdampslist:
		key, start, stop = amp
		xrt_0[key] = np.concatenate([xrt_0[key][:start],np.interp(np.arange(stop-start),[0,stop-start],[xrt_0[key][start],xrt_0[key][stop]]),xrt_0[key][stop:]])

	line = 3910
	here1 = np.where((wave>line-10)&(wave<line+10))[0]
	line = 4359
	here2 = np.where((wave>line-10)&(wave<line+10))[0]
	line = 5461
	here3 = np.where((wave>line-10)&(wave<line+10))[0]
	if SMOOTHATA:
		for key in xrt_0.keys():
			here = here1
			slope = (xrt_0[key][here[-1]+1] - xrt_0[key][here[0]-1])/float(len(here))
			xrt_1 = np.concatenate([xrt_0[key][:here[0]], xrt_0[key][here[0]-1] + np.arange(len(here))*slope, xrt_0[key][here[-1]+1:]])
			here = here2
			slope = (xrt_0[key][here[-1]+1] - xrt_0[key][here[0]-1])/float(len(here))
			xrt_1 = np.concatenate([xrt_0[key][:here[0]], xrt_0[key][here[0]-1] + np.arange(len(here))*slope, xrt_0[key][here[-1]+1:]])
			here = here3
			slope = (xrt_0[key][here[-1]+1] - xrt_0[key][here[0]-1])/float(len(here))
			xrt_1 = np.concatenate([xrt_0[key][:here[0]], xrt_0[key][here[0]-1] + np.arange(len(here))*slope, xrt_0[key][here[-1]+1:]])
			xrt[key] = interp1d(wave, gaussian_filter(xrt_1, sigma=SIGMA), fill_value='extrapolate')
	else:
		for key in xrt_0.keys():
			xrt[key] = interp1d(wave, xrt_0[key], fill_value='extrapolate')
	return xrt

def boxes(array, flag, size):
	boxes2 = np.split(np.arange(112), size[0])
	boxes1 = np.split(np.arange(1032), size[1])
	newarray = np.array(array.copy(), dtype=np.float64)/np.array(array.copy())
	medfilt = np.zeros(array.shape)
	medians = np.zeros(size)
	for i, box1 in enumerate(boxes2):
		for j, box2 in enumerate(boxes1):
			medians[i,j] = np.nanmedian(array[box1][:,box2][flag[box1]])
			for k in box1:
				for l in box2:
					medfilt[k,l] = medians[i,j]
	return gaussian_filter(medfilt, sigma=(8,100))


# SWITCHES

SIGMA = 4
SMOOTHATA = SIGMA != 0
THRESHOLD = 1.03
LOWER = 0.96
ADJUSTMENT = THRESHOLD == 0
FILTER = True
KAPPA = 2.7

def_wave = np.arange(3470., 5542., 2.)
filename = '/work/03946/hetdex/hdr1/reduction/data/{}.h5'.format(shot)
fileh = tb.open_file(filename, 'r')

table =  Table(fileh.root.Data.Fibers.read())

expnum = table["expnum"].data
sky_spectra_orig = table["spectrum"].data
sky_spectra_orig = np.array(sky_spectra_orig, dtype=np.float64)
sky_subtracted_orig = table["sky_subtracted"].data
sky_subtracted_orig = np.array(sky_subtracted_orig, dtype=np.float64)#[expnum==1]
fiber_to_fiber_orig = table["fiber_to_fiber"].data
fiber_to_fiber_orig = np.array(fiber_to_fiber_orig, dtype=np.float64)#[expnum==1]
wavelength_orig = table["wavelength"].data
wavelength_orig = np.array(wavelength_orig, dtype=np.float64)#[expnum==1]

ifuslots = table["ifuslot"].data
ifuslots = np.array([x[0] for x in np.split(ifuslots, ifuslots.shape[0]/112)])

multinames = table['multiframe'].data
multinames = np.array([x[0] for x in np.split(multinames,multinames.shape[0]/112)])

amps = table["amp"].data
amps = np.array([x[0] for x in np.split(amps, amps.shape[0]/112)])

exposures = np.array(["exp0{}".format(x) for x in table["expnum"].data])
exposures = np.array([x[0] for x in np.split(exposures, exposures.shape[0]/112)])

rescor = get_rescor(ifuslots, amps, def_wave)

meanmean = np.nanmedian(sky_spectra_orig[:,600:900], axis=1)
biw = biweight_location(meanmean)
biscale = biweight_scale(meanmean)
if False:
	flag = meanmean < biw + KAPPA*biscale
else:
	flag = meanmean == meanmean

sky_spectra = np.array(np.split(sky_spectra_orig, sky_spectra_orig.shape[0]/112))[exposures==exp]
sky_subtracted = np.array(np.split(sky_subtracted_orig, sky_subtracted_orig.shape[0]/112))[exposures==exp]
fiber_to_fiber = np.array(np.split(fiber_to_fiber_orig, fiber_to_fiber_orig.shape[0]/112))[exposures==exp]
wavelength = np.array(np.split(wavelength_orig, wavelength_orig.shape[0]/112))[exposures==exp]

ifuslots = ifuslots[exposures==exp]
amps = amps[exposures==exp]
multinames = multinames[exposures==exp]

flag = np.array(np.split(flag, flag.shape[0]/112))[exposures==exp]

xrt = get_xrt_new()

sky_spectra_iter = sky_spectra.copy()
print('sky_spectra_iter.shape ; ', sky_spectra_iter.shape)
for i in range(sky_spectra_iter.shape[0]):
	try:
		sky_spectra_iter[i] = sky_spectra_iter[i]/(xrt[(ifuslots[i], amps[i])](wavelength[i])*fiber_to_fiber[i])
	except KeyError:
		sky_spectra_iter[i] *= 0

binsize = int(sky_spectra_iter[flag].size/3000.)
nwave_iter, smooth_iter = make_avg_spec(wavelength[flag], sky_spectra_iter[flag], binsize=binsize)
#nwave, smooth = get_common_sky(sky_spectra, fiber_to_fiber, wavelength)

csm_iter = interp1d(nwave_iter, smooth_iter, fill_value="extrapolate")

# get new sky as csm*xrt*f2f*factor
ata_adj = []
new_sky_iter = sky_spectra_iter.copy()
for i in range(new_sky_iter.shape[0]):
	try:
		factor = np.nanmedian(new_sky_iter[i]/csm_iter(wavelength[i]))
		if factor > THRESHOLD:
			factor = THRESHOLD
		elif factor < LOWER:
			factor = LOWER
		ata_adj.append(factor)
		new_sky_iter[i] = csm_iter(wavelength[i]) * xrt[(ifuslots[i], amps[i])](wavelength[i]) * factor * fiber_to_fiber[i]
	except KeyError:
		new_sky_iter[i] *= 0
ata_adj = np.array(ata_adj)

xskysub_iter = []
medianfilters = []
N = len(sky_spectra)

START, STOP = 0,N

for i in range(len(sky_spectra))[START:STOP]:
	new_skysub = sky_spectra[i] - new_sky_iter[i]*(1+rescor[(ifuslots[i], amps[i])])
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
	#if i%10==0:
	#	print( "done {} / {} ".format(i, N))
print('done with first skysub.\n')
xskysub_iter = np.array(xskysub_iter)

first_sky_subtracted = np.concatenate(xskysub_iter)

meanmean = np.nanmedian(first_sky_subtracted[:,600:900], axis=1)
biw = biweight_location(meanmean)
biscale = biweight_scale(meanmean)

if True:
	flag = meanmean < biw + KAPPA*biscale
else:
	flag = meanmean == meanmean

newflagged = []

# flag the fiber before and after a flagged fiber
if True:
	for i in range(len(flag)):
		if not flag[i]:
			if i in newflagged:
				continue
			else:
				if i%112==0& i!=0 & flag[i+1]:
					flag[i+1] = False
					newflagged.append(i+1)
				elif i%112==111 & flag[i-1]:
					flag[i-1] = False
					newflagged.append(i-1)
				elif i == 0 :
					if flag[i+1]:
						flag[i+1] = False
						newflagged.append(i+1)
				elif i!=flag.size-1:
					if flag[i-1]:
						flag[i-1] = False
						newflagged.append(i-1)
					if flag[i+1]:
						flag[i+1] = False
						newflagged.append(i+1)


sky_spectra = np.array(np.split(sky_spectra_orig, sky_spectra_orig.shape[0]/112))[exposures==exp]
sky_subtracted = np.array(np.split(sky_subtracted_orig, sky_subtracted_orig.shape[0]/112))[exposures==exp]
fiber_to_fiber = np.array(np.split(fiber_to_fiber_orig, fiber_to_fiber_orig.shape[0]/112))[exposures==exp]
wavelength = np.array(np.split(wavelength_orig, wavelength_orig.shape[0]/112))[exposures==exp]

flag = np.array(np.split(flag, flag.shape[0]/112))#[exposures==exp]

sky_spectra_iter = sky_spectra.copy()
sky_spectra_iter.shape

for i in range(sky_spectra_iter.shape[0]):
	try:
		sky_spectra_iter[i] = sky_spectra_iter[i]/(xrt[(ifuslots[i], amps[i])](wavelength[i])*fiber_to_fiber[i]*ata_adj[i])
	except KeyError:
		sky_spectra_iter[i] *= 0

binsize = int(sky_spectra_iter[flag].size/3000.)
print(binsize)
nwave_iter, smooth_iter = make_avg_spec(wavelength[flag], sky_spectra_iter[flag], binsize=binsize)
#nwave, smooth = get_common_sky(sky_spectra, fiber_to_fiber, wavelength)

csm_iter = interp1d(nwave_iter, smooth_iter, fill_value="extrapolate", kind='cubic') # changed interpolation to cubic

def gaus(x,a,x0,offset):
	return offset+a*np.exp(-(x-x0)**2/(2*2.2**2))

WAVELENGTHSHIFT = True
WINDOWSIZE = 30.
if WAVELENGTHSHIFT:
	# TRY: wavelength shift
	x, y = nwave_iter[(nwave_iter>=5461.4-WINDOWSIZE)&(nwave_iter<=5461.4+WINDOWSIZE)], csm_iter(nwave_iter[(nwave_iter>=5461.4-WINDOWSIZE)&(nwave_iter<=5461.4+WINDOWSIZE)])

	mean = sum(x * y) / sum(y)
	#sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))

	popt,pcov = curve_fit(gaus,x,y,p0=[300,mean,200])

	csm_mid = popt[1]


# get new (iterated) sky as csm * xrt * f2t * factor
final_adj=[]
new_sky_iter = sky_spectra_iter.copy()
triplesave = []
wlshifts = []
for i in range(new_sky_iter.shape[0]):
	if WAVELENGTHSHIFT:
		try:
			binsize=int(sky_spectra_iter[i][flag[i]].size/3000.)
			nwave_tmp, smooth_tmp = make_avg_spec(wavelength[i][flag[i]], sky_spectra_iter[i][flag[i]], binsize=35, knots=None)
			nwave_tmp, smooth_tmp = np.array(nwave_tmp), np.array(smooth_tmp)
			x, y = nwave_tmp[(nwave_tmp>=5461.4-WINDOWSIZE)&(nwave_tmp<=5461.4+WINDOWSIZE)], smooth_tmp[(nwave_tmp>=5461.4-WINDOWSIZE)&(nwave_tmp<=5461.4+WINDOWSIZE)]
			mean = sum(x * y) / sum(y)
			#sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))
			popt,pcov = curve_fit(gaus,x,y,p0=[300,mean,200])
			this_mid = popt[1]
			diff = this_mid - csm_mid
			print diff
			wlshifts.append(diff)
		except Exception as e:
			wlshifts.append(0.0)
			print e
			new_sky_iter[i] = csm_iter(wavelength[i]) * xrt[(ifuslots[i], amps[i])](wavelength[i]) * factor * fiber_to_fiber[i] #* ata_adj[i]
	
	try:
		if flag[i][flag[i]].size > 30:
			factor = np.nanmedian(new_sky_iter[i][flag[i]]/csm_iter(wavelength[i][flag[i]]))
			div = new_sky_iter[i][flag[i]]/csm_iter(wavelength[i][flag[i]])
			triplesave.append([np.nanmedian(div[:,:350]),np.nanmedian(div[:,350:700]),np.nanmedian(div[:,700:])])
		else:
			factor = 1
			triplesave.append([1.,1.,1.])
		factor *= ata_adj[i]
		if factor > THRESHOLD:
			factor = THRESHOLD
		elif factor < LOWER:
			factor = LOWER
		final_adj.append(factor)
		
		if WAVELENGTHSHIFT:
			new_sky_iter[i] = csm_iter(wavelength[i]-diff) * xrt[(ifuslots[i], amps[i])](wavelength[i]) * factor * fiber_to_fiber[i] #* ata_adj[i]
		else:
			new_sky_iter[i] = csm_iter(wavelength[i]) * xrt[(ifuslots[i], amps[i])](wavelength[i]) * factor * fiber_to_fiber[i] #* ata_adj[i]
	except KeyError:
		triplesave.append([0.,0.,0.])
		new_sky_iter[i] *= 0
triplesave = np.array(triplesave)
wlshifts = np.array(wlshifts)
ascii.write(Table({'ifuslot':ifuslots,'amp':amps, ':350':triplesave[:,0], '350:700':triplesave[:,1], '700:':triplesave[:,2]}), '/work/05865/maja_n/stampede2/a2a_adj/{}-{}-a2a_adj.dat'.format(shot, exp), overwrite=True)
print('wrote /work/05865/maja_n/stampede2/a2a_adj/{}-{}-a2a_adj.dat'.format(shot, exp))
final_adj = np.array(final_adj)

for i in np.where((ifuslots=='013'))[0]:
    for j in range(112):
        new_sky_iter[i,j] = gaussian_filter(new_sky_iter[i,j], sigma=0.3*2)

order = np.argsort(ifuslots)

# subtract final sky
second_skysub_iter = []
medianfilters = []
fiber_to_fiber_int, new_sky_iter_int = [], []
N = len(sky_spectra)

size = (8,8)

relres = []
INTERPOLATE = True
REBIN = False
orig_rebin = []
for i in order:#range(len(sky_spectra))[START:STOP]:
	new_skysub = (sky_spectra[i] - new_sky_iter[i]*(1+rescor[(ifuslots[i], amps[i])]))
	if FILTER:
		medfilt = boxes(new_skysub, flag[i], size=size)#newboxcar(new_skysub, flag[i], FILTERSIZE) #median_filter(new_skysub[flag[i]], size=medfilt_size)
		medfilt[~np.isfinite(medfilt)] = 0
		medianfilters.append(medfilt)
		new_skysub -= medfilt
	new_skysub_rel  = new_skysub/(new_sky_iter[i]*(1+rescor[(ifuslots[i], amps[i])]))

	primhdu = fits.PrimaryHDU(new_skysub_rel)
	hdul = fits.HDUList([primhdu])
	hdul.writeto("/work/05865/maja_n/stampede2/rescor/tmp/rc_"+shot+"_"+exp+"_"+multinames[i]+".fits", overwrite=True)
	print("wrote to /work/05865/maja_n/stampede2/rescor/tmp/rc_"+shot+"_"+exp+"_"+multinames[i]+".fits")
	new_skysub_int = []
	#orig_int = []
	#new_skysub_int_rel = []
	#f2f_int, nsi_int = [], []
	new_skysub[sky_spectra[i]==0] = 0
	#new_skysub_rel[sky_spectra[i]==0] = 0
	if INTERPOLATE:
		for j in range(112):
 			dw = np.diff(wavelength[i,j])
 			dw = np.hstack((dw[0], dw))
			#dw = 1
			new_skysub_int.append(np.interp(def_wave, wavelength[i,j], new_skysub[j]/dw, left=0.0, right=0.0))
			#new_skysub_int_rel.append(np.interp(def_wave, wavelength[i,j], new_skysub_rel[j]/dw, left=0.0, right=0.0))
			#f2f_int.append(np.interp(def_wave, wavelength[i,j], fiber_to_fiber[i,j], left=1., right=1.))
			#nsi_int.append(np.interp(def_wave, wavelength[i,j], new_sky_iter[i,j]/dw, left=0.0, right=0.0))		
	elif REBIN:	
		ww, rebinned_skysub = get_rebinned(i,new_skysub)
		new_skysub_int = rebinned_skysub
		#new_skysub_int_rel.append(get_rebinned(new_skysub_rel[i]))
		f2f_int = get_rebinned(i,fiber_to_fiber[i])[1]
		nsi_int = get_rebinned(i,new_sky_iter[i])[1]
		orig_int = get_rebinned(i, sky_subtracted[i])[1]
	
	second_skysub_iter.append(new_skysub_int)
	#relres.append(new_skysub_int_rel)
	#fiber_to_fiber_int.append(f2f_int)
	#new_sky_iter_int.append(nsi_int)
	#orig_rebin.append(orig_int)

	print( "done {} / {} ".format(i, N))
second_skysub_iter = np.array(second_skysub_iter)
#orig_rebin = np.array(orig_rebin)
print("second Skysub iter.shape : ", second_skysub_iter.shape)
#relres = np.array(relres)
#fiber_to_fiber_int, new_sky_iter_int = np.array(fiber_to_fiber_int), np.array(new_sky_iter_int)

SAVEASFITS = True
if SAVEASFITS:
	for i in range(len(order)):
		idx = order[i]
		name = multinames[idx]
		thisskysub = second_skysub_iter[i]
		#thissky = new_sky_iter[idx]*(1+rescor[(ifuslots[idx],amps[idx])])
		header = fits.Header()
		header['wavelength_shift'] = wlshifts[idx]
		hdu = fits.PrimaryHDU(thisskysub, header=header)
		#hdu2 = fits.ImageHDU(thissky, name="sky_spectrum")
		hdulist = fits.HDUList([hdu])
		filename = '/work/05865/maja_n/stampede2/ffskysub/{}/{}/{}.fits'.format(shot,exp,name)
		hdulist.writeto(filename, overwrite=True)
		print('wrote to '+filename)



tickarray = np.array(np.split((np.arange(flag.size)), flag.shape[0]))[START:STOP][np.where(~flag[order])]

ampticks = np.arange(N)*112

PLOT = False
PCAPLOT = False
if PLOT:
	plt.figure(figsize=(25,200))
	plt.subplot(131)
	plt.title("original")
	plt.imshow(np.concatenate(sky_subtracted[order]), vmin=-40, vmax=40, interpolation="none", aspect="auto", cmap="Greys_r")
	plt.yticks((tickarray - START*112), 'x'*len(tickarray), color='red'); #- START*112
#	plt.colorbar();
	plt.subplot(132)
	plt.title("xskysub")
	plt.imshow(np.concatenate(second_skysub_iter), vmin=-40, vmax=40, interpolation="none", aspect="auto", cmap="Greys_r")
	plt.yticks(ampticks, [ifuslots[i]+amps[i] for i in order], rotation=90)#(tickarray - START*112, 'x'*len(tickarray), color='red')
#	plt.colorbar();
	plt.subplot(133)
	plt.title("original - xskysub")
	#plt.imshow(np.concatenate(relres), vmin=-0.02, vmax=0.02, interpolation="none", aspect="auto", cmap="Greys_r");
	plt.imshow(np.concatenate(orig_rebin-second_skysub_iter), vmin=-10, vmax=10, interpolation="none", aspect="auto", cmap="Greys_r");
	plt.yticks(ampticks, [ifuslots[i]+amps[i] for i in order], rotation=90);
#	plt.colorbar(orientation="horizontal");""""""
	#plt.savefig('../lsspresent/test-fullframe-{}-{}-own-smooth-skip-shift.png'.format(shot, exp), bbox_inches='tight')
	plt.savefig('../lsspresent/original-ffskysub-{}-{}.png'.format(shot, exp), bbox_inches='tight')


	original_sky_spectra = table["sky_spectrum"].data
	original_sky_spectra = np.array(np.split(original_sky_spectra, original_sky_spectra.shape[0]/112))[exposures==exp][order]
	print original_sky_spectra.shape
	plt.figure(figsize=(25,200))
	plt.subplot(131)
	plt.title("original fiber2fiber")
	plt.imshow(np.concatenate(fiber_to_fiber[order]), vmin=-1, vmax=1, interpolation="none", aspect="auto", cmap="Greys_r")
	plt.yticks((tickarray - START*112), 'x'*len(tickarray), color='red'); #- START*112
#	plt.colorbar();
	plt.subplot(132)
	plt.title("rebinned fiber2fiber")
	plt.imshow(np.concatenate(fiber_to_fiber_int), vmin=-1, vmax=1, interpolation="none", aspect="auto", cmap="Greys_r")
	plt.yticks(ampticks, [ifuslots[i]+amps[i] for i in order], rotation=90)#(tickarray - START*112, 'x'*len(tickarray), color='red')
#	plt.colorbar();
	plt.subplot(133)
	plt.title("original sky_spectrum")
	plt.imshow(np.concatenate(original_sky_spectra), vmin=-40, vmax=200, interpolation="none", aspect="auto", cmap="Greys_r")
	#plt.imshow(np.concatenate(relres), vmin=-0.02, vmax=0.02, interpolation="none", aspect="auto", cmap="Greys_r")
	plt.subplot(143)
	plt.title("rebinned ff sky_spectrum")
	plt.imshow(np.concatenate(new_sky_iter_int), vmin=-40, vmax=200, interpolation="none", aspect="auto", cmap="Greys_r");
	plt.yticks(ampticks, [ifuslots[i]+amps[i] for i in order], rotation=90);
#	plt.colorbar(orientation="horizontal");""""""
	plt.savefig('../lsspresent/fiber2fiber_sky_spectrum_{}_{}.png'.format(shot, exp), bbox_inches='tight')


if PCAPLOT:
	# compare to fullframe pca
	fullframepca = pickle.load(open('../fullframe-{}-{}-pca.pickle'.format(shot, exp),'rb'))
	pcaarray = []
	for i in order:
		try:
			pcaarray.append(fullframepca[(ifuslots[i], amps[i])])
		except KeyError as e:
			pcaarray.append(np.zeros((112,1010)))
			print e
	pcashow = np.concatenate(pcaarray)

	plt.figure(figsize=(25,200))
	plt.subplot(131)
	plt.title("original")
	plt.imshow(np.concatenate(sky_subtracted[order]), vmin=-40, vmax=40, interpolation="none", aspect="auto", cmap="Greys_r")
	plt.yticks((tickarray - START*112), 'x'*len(tickarray), color='red'); #- START*112
#	plt.colorbar();
	plt.subplot(132)
	plt.title("xskysub")
	plt.imshow(np.concatenate(second_skysub_iter), vmin=-20, vmax=20, interpolation="none", aspect="auto", cmap="Greys_r")
	plt.yticks(ampticks, [ifuslots[i]+amps[i] for i in order], rotation=90)#(tickarray - START*112, 'x'*len(tickarray), color='red')
#	plt.colorbar();
	plt.subplot(133)
	plt.title("pca skysub")
	plt.imshow(pcashow, vmin=-40, vmax=40, interpolation="none", aspect="auto", cmap="Greys_r");
	plt.yticks(ampticks, [ifuslots[i]+amps[i] for i in order], rotation=90);
#	plt.colorbar(orientation="horizontal");""""""
	plt.savefig('../lsspresent/test-fullframe-{}-{}-own-smooth-skip-pca.png'.format(shot, exp), bbox_inches='tight')

SAVEASPICKLE = False
if SAVEASPICKLE:
	for j,i in enumerate(order):
		thissubtracted = second_skysub_iter[j]
		thisifu, thisamp = ifuslots[i], amps[i]
		filename = open('../skysubtractedfiles/{}/{}/xsubsr_multi_xxx_{}_xxx_{}_rebin.pickle'.format(shot, exp, thisifu, thisamp),'wb')
		if REBIN:
			pickle.dump((ww, {'sky_subtracted':thissubtracted,'sky_spectrum':new_sky_iter_int[j],'fiber_to_fiber':fiber_to_fiber_int[j], "originalskysub":orig_rebin[j]}), filename)
		elif INTERPOLATE:
			pickle.dump((def_wave, {'sky_subtracted':thissubtracted,'sky_spectrum':new_sky_iter_int[j],'fiber_to_fiber':fiber_to_fiber_int[j], "originalskysub":orig_rebin[j]}), filename)
		print('wrote to ../skysubtractedfiles/{}/{}/xsubsr_multi_xxx_{}_xxx_{}_rebin.pickle'.format(shot, exp, thisifu, thisamp))

