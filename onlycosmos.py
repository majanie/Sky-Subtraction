import glob
import pickle
import numpy as np
import os
from astropy.io import fits
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def gauss(a,b,c,x):
    return a*np.exp(-(x-b)**2/c)**2

def load_skys(ff,which="sky_spectrum"):
    skys = {}
    shotids = []
    N = len(ff)
    sff = []
    for i,f in enumerate(ff):
        if i % 100 == 0:
            print("loading {} out of {}.".format(i,N))
        shotid = f.split("/")[6]
        exp = f.split("/")[7]
        try:
            ww,rb = pickle.load( open(f,'rb'), encoding='iso-8859-1' )
            skys[(shotid,exp)] = rb[which]/rb["fiber_to_fiber"]
            sff.append(f)
        except:
            print("Error loading {}.".format(f))
            pass
    print("starting wl = ", ww[0], "A")
    return ww, skys, sff

def load_skys2(ff,which="sky_spectrum"):
    skys = {}
    shotids = []
    N = len(ff)
    sff = []
    for i,f in enumerate(ff):
        if i % 100 == 0:
            print("loading {} out of {}.".format(i,N))
        print(f)
        shotid = f.split("/")[6]
        exp = f.split("/")[7]
        ifu = f.split("/")[8][10:13]
        amp = f.split("/")[8][18:20]
        print(shotid, exp, ifu, amp)
        try:
            ww,rb = pickle.load( open(f,'rb'), encoding='iso-8859-1' )
            skys[(shotid,exp,ifu,amp)] = rb[which]/rb["fiber_to_fiber"]
            sff.append(f)
        except:
            print("Error loading {}.".format(f))
            pass
    print("starting wl = ", ww[0], "A")
    return ww, skys, sff

cosmosshots = ["20180110v021","20180113v013","20180114v013","20180120v008","20180123v009","20180124v010","20180209v009","20180210v006"]
""" cosmos shots
20171220v015
20171221v016
20171222v013
20171225v015

---

20180110v021
20180113v013
20180114v013
20180120v008
20180123v009
20180124v010
20180209v009
20180210v006"""

# lies all fuer amp LL in IFU 22
ff_022_LL1= glob.glob("/data/hetdex/u/mxhf/rebin/201801??v???/exp0?/multi_???_???_???_LL_rebin.pickle")
ff_022_LL2= glob.glob("/data/hetdex/u/mxhf/rebin/201802??v???/exp0?/multi_???_???_???_LL_rebin.pickle")
ff_022_LL3= glob.glob("/data/hetdex/u/mxhf/rebin/201803??v???/exp0?/multi_???_???_???_LL_rebin.pickle")

ff_022_LL = np.concatenate([ff_022_LL1,ff_022_LL2,ff_022_LL3])

skys = {}
ww,skys[("022","LL")],sff = load_skys2(ff_022_LL,which="sky_spectrum")
#print(skys[("022","LL")], skys[('022','LL')][list(skys[('022','LL')].keys())[0]].shape)

fiberarray = np.concatenate([x[:,:] for x in skys[('022','LL')].values()])
allkeys = np.array(list(skys[('022','LL')].keys())) # Ich hoffe, die Reihenfolgen bleiben gleich...

print('fiberarray.shape: ' , fiberarray.shape)

fiberarray[np.where(np.isnan(fiberarray))] = 0

#fiberarray.shape

meanspec = np.nanmean(fiberarray[:], axis=0)
stdspec = np.nanstd(fiberarray[:],axis=0)

fiberarray_0 = (fiberarray-meanspec)/stdspec

#stdspec.shape
#plt.figure(figsize=(20,4))
#plt.plot(meanspec)
#plt.title('mean spectrum')
#plt.show()

cov = np.cov(fiberarray_0.T)

eigenvals, eigenvecs = np.linalg.eig(cov)
eigenvals = np.real(eigenvals)
eigenvecs = np.real(eigenvecs)

eigenpairs = [(np.abs(eigenvals[i]), eigenvecs[:,i]) for i in np.argsort(abs(eigenvals))[::-1]]
ordered_eigenvecs = np.array([eigenpairs[i][1] for i in range(len(eigenvals))])
"""plt.figure(figsize=(20,4))
plt.plot([x[0] for x in eigenpairs])
plt.axvline(100)
plt.axvline(150)
plt.yscale('log')"""

ncomp = 100
imp = ordered_eigenvecs[:ncomp]

onlycosmos = []
onlycosmoskeys = []

for i in range(allkeys.shape[0]):
    key = allkeys[i]
    if key[0] in cosmosshots:
        tmp = (skys[('022','LL')][tuple(key)]-meanspec)/stdspec
        onlycosmos.append(tmp)#(fiberarray_0[i:i+112])
        onlycosmoskeys.append(key)
onlycosmos = np.array(onlycosmos)
print('shape onlycosmos before: ', onlycosmos.shape)
onlycosmos, onlycosmoskeys = np.concatenate(onlycosmos), np.array(onlycosmoskeys)
print('shape onlycosmos: ', onlycosmos.shape)
print('onlycosmoskeys: ', onlycosmoskeys)

imp = ordered_eigenvecs[:ncomp]
#print('imp[np.isfinite(imp)].shape',imp[np.isfinite(imp)].shape)

fiberpca = np.dot(onlycosmos, imp.T)
#print('fiberpca[np.isfinite(fiberpca)].shape ',fiberpca[np.isfinite(fiberpca)].shape)

newspec = np.dot(fiberpca, imp)
#print('new[np.isfinite(new)].shape ',new[np.isfinite(new)].shape)

newspec = newspec*stdspec+meanspec
#print('fiberpca[np.isfinite(newspec)].shape ',newspec[np.isfinite(newspec)].shape)

onlycosmos = onlycosmos*stdspec+meanspec
re = (onlycosmos-newspec)/onlycosmos
#print('fiberpca[np.isfinite(re)].shape ',re[np.isfinite(re)].shape)

onlycosmos = [onlycosmos[i:i+112,:] for i in range(0,onlycosmos.shape[0],112)]
onlycosmos = np.array(onlycosmos)
print("(onlycosmos.shape)", onlycosmos.shape)

newspec = [newspec[i:i+112,:] for i in range(0,newspec.shape[0],112)]
newspec = np.array(newspec)
print("(newspec.shape)", newspec.shape)

myindex = [-1]
stddevi = np.nanstd(re[myindex])
print('\nstd of re[myindex]: ',stddevi, '\nncomp = ', ncomp)

for i in range(onlycosmos.shape[0]):
    print(onlycosmoskeys[i])
    fin = "pcaskies/pcasky_multi_{}_{}_{}_{}.fits".format(onlycosmoskeys[i][0],onlycosmoskeys[i][1],onlycosmoskeys[i][2],onlycosmoskeys[i][3])  # this is specfic for the 20180822 shots... CHANGE IT!!!
    hdul = fits.HDUList([fits.PrimaryHDU(skys[('022','LL')][tuple(onlycosmoskeys[i])]),fits.ImageHDU(newspec[i], name='pcasky')])
    hdul.writeto(fin, overwrite=True)
    print('Wrote {}'.format(fin))

#for i in range(len(onlycosmos)):
#    fin = gg[i]  # this is specfic for the 20180822 shots... CHANGE IT!!!
#    hdul = fits.HDUList([fits.ImageHDU(allrescaled[i], name='pcasky')])
#    hdu.writeto('{}_pcasky'.format(fin.split('/')[-1]), overwrite=True)
#    print('Wrote {}'.format(fin))

