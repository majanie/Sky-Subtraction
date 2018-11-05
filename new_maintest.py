import sys
import os
import glob
import numpy as np
from astropy.io import fits

import new as xsky
import new_rebinning as rebinning

def main():
  
	shots = ['20180124v010','20180124v011']
	#['20180210v006','20180210v005','20180210v007']
	#['20180209v008','20180209v009','20180209v010']
	#['20180124v010','20180124v011']
	#['20180123v008','20180123v009','20180123v010']
	#['20180114v013','20180114v014','20180114v012']
	#['20180113v012','20180113v013','20180113v014']
	#['virus0000006', 'virus0000010', 'virus0000011', 'virus0000012', 'virus0000013', 'virus0000014', 'virus0000015']
	#['20180120v008','20180120v009']
	#['20180110v021', '20180110v017','20180110v016'] 
	#['20180110v007', '20180110v008', '20180110v009', '20180110v016', '20180110v017','20180110v021'] 
	
	flagged_ifus = {'20180110v021':['36','42','43','74','86','95','104'],
			'20180120v008':['36','42','43','74','86','95','104'],
			'20180123v009': ['36','42','43','74','86','95','104']}#,
		#	'20180124v010':['36','42','43','74','86','95','104']}

	# should the new fits files with xskysub extension replace the 'old' ones
	overwrite = False

	# if overwrite == false, please propose a name for the new fits files.
	# it will look like 'name_shot_exp_ifuslot_amp.fits' in your current directory
	name = None 
	
	xskysub( shots = shots, flagged_ifus = flagged_ifus, overwrite = overwrite)

	return 1	

def xskysub(shots, flagged_ifus, overwrite, name = None):

	# extract shots, exposures, ifuslots and amplifiers from file name
	night = shots[0][:-4] 
	pattern = '/work/03946/hetdex/maverick/red1/reductions/{}/virus/virus0000*/exp0?/virus/multi*_042*U.fits'.format(night)
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
	ee, ss, aa, shs = np.sort(np.unique(exposures)), np.sort(np.unique(ifuslots)), np.sort(np.unique(amps)), np.sort(np.unique(shots)) 
	goodshotmask = np.ones(len(shs), dtype=np.int8)*True
	# only use shots with more than one exposure, i.e. hetdex shots
	for shot in shots:
	#for shot in np.unique(allshots):
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
			goodshotmask[np.where(shs==shot)] = False
			print('{} has no sky spectrum'.format(shot))
	
	shots = shs[np.where(goodshotmask)]
	shots = np.array(shots)
	print('These shots are going to be used: ', shots)
	
	# rebin spectrum, sky_spectrum and fiber_to_fiber extensions of every amplifier in every shot
	# compute xsky and subtract from spectrum (rebinned)
	# rel_error is the difference between xsky spectrum and sky spectrum (both rebinned), array

	rebinned = {}

	for shot in shots:
		pattern2 = '/work/03946/hetdex/maverick/red1/reductions/{}/virus/virus0000{}/exp0?/virus/multi*_042*U.fits'.format(night, shot[-3:]) # pattern2, because we need pattern later
		ww, rebinned[shot] = rebinning.main(pattern2, night=shot[:-4], shot=shot[-3:], extensions = ['sky_spectrum', 'fiber_to_fiber','spectrum'])
	print('Rebinned :)')
	xsky_spectrum, xsky_sub, rel_error = xsky.main(shots, ww, rebinned, flagged_ifus)
	print('xsky subtracted \(^-^)/')
	# write xsky subtracted extension to fits file
	name = 'sometestfitsname'	# not neccessary at the moment
	for shot in shots:
		for exp in ee:
			for ifuslot in ss: # when flagged, as easy, since this is the part that adds the xsysub to multifits
				for amp in aa:

					ii = (exposures == exp) * (ifuslots == ifuslot) * (amps == amp) *  (allshots == 'virus0000{}'.format(shot[-3:]))
					if not sum(ii) == 1:
						print("Error: Could not identify file for exposure {} slot {} and amp {}.".format(exp, ifuslot, amp))
						continue
                       
					fin = ff[ii][0]
					print("Writing {}".format(fin))
					hdu = fits.open(fin)
					try:
						hdu.append(fits.ImageHDU( xsky_sub[(shot,(exp, ifuslot, amp))], name='xsky_subtracted'))
					#hdu.append(fits.ImageHDU( rebinned[shot][(exp, ifuslot, amp)]['fiber_to_fiber'], name='rbfiber'))
					#hdu.append(fits.ImageHDU( rebinned[shot][(exp, ifuslot, amp)]['sky_spectrum'], name='rbsky'))
					#hdu.append(fits.ImageHDU( rebinned[shot][(exp, ifuslot, amp)]['spectrum'], name='rbspectrum'))
					#hdu.append(fits.ImageHDU( xsky_spectrum[shot][(exp, ifuslot, amp)], name='xsky'))
						hdu.append(fits.ImageHDU( rel_error[(shot,(exp, ifuslot, amp))], name='rel_error'))

						if overwrite:
							hdu.writeto(fin, overwrite = True)
						else: # Please make sure you have enough disk space in your current directory...
							hdu.writeto('test{}_{}_{}'.format(shot, exp, fin.split('/')[-1]), overwrite=True)
						hdu.close()
					except KeyError:
						hdu.close()
						print("Could not write to file {}".format(fin))
						pass

	hdu = fits.PrimaryHDU(np.arange(0,1,2))
	error = fits.HDUList([hdu])
	error.append(fits.ImageHDU(np.array(list(rel_error.values()))))
	error.writeto('testerror_{}'.format(night), overwrite=True)

        # maybe print what the program is doing...?

	return 1

#main()
