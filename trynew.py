import glob
import pickle
import numpy as np
import os
from sklearn.decomposition import PCA

"""Simple PCA using all rebinned sky spectra in 2018. For each IFU (all amplifiers), PCA (i.e. dimensionality reduction)
with n = 20 components is applied and the resulting sky spectra for the cosmos shots are saved as pickle files."""

def load_skys(ff,which="sky_spectrum"): # loads rebinned sky spectra divided by fiber to fiber from the pickle files
    skys = {}
    shotids = []
    N = len(ff)
    sff = []
    for i,f in enumerate(ff):
        if i % 100 == 0:
            print("loading {} out of {}.".format(i,N))
        #print(f)
        shotid = f.split("/")[6]
        exp = f.split("/")[7]
        ifu = f.split("/")[8][10:13]
        amp = f.split("/")[8][18:20]
        #print(shotid, exp, ifu, amp)
        try:
            ww,rb = pickle.load( open(f,'rb'), encoding='iso-8859-1' )
            skys[(shotid,exp,ifu,amp)] = rb[which]/rb["fiber_to_fiber"]
            sff.append(f)
        except:
            print("Error loading {}.".format(f))
            pass
    print("starting wl = ", ww[0], "A")
    return ww, skys, sff

def save_sky(IFU, amp , k, pca_sky): # saves new pca sky spectra in the pickle files #/data/hetdex/u/mxhf/rebin/20170331v006/exp01
    pattern='/data/hetdex/u/mxhf/rebin/{}/{}/multi_???_{}_???_{}_rebin.pickle' #"pca_test/rebin/{}/{}/multi_???_{}_???_{}_rebin.pickle"
    shotid, exp = k

    _pattern = pattern.format(shotid, exp, IFU, amp)
    ff = glob.glob(_pattern)
    if not len(ff) == 1:
        print("ERROR: Did not find files like {}".format(_pattern))
        return
    fname = ff[0]

    h,t = os.path.split(fname)
    #pca_fname = os.path.join(h,"pca_" + t)
    h2 = 'tmp/{}/{}'.format(shotid, exp)
    pca_fname = os.path.join(h2,"pca_" + t)

    ww,rb = pickle.load( open(fname,'rb'), encoding='iso-8859-1' )
    #rb["fiber_to_fiber"] = rb["fiber_to_fiber"][:,:N]
    rb["pca_sky_spectrum"] = rb["sky_spectrum"].copy()
    #b["pca_sky_spectrum"][:,:N][:,ii] = pca_sky * rb["fiber_to_fiber"][:,:N][:,ii]
    rb["pca_sky_spectrum"] = pca_sky * rb["fiber_to_fiber"]
    #print('\nshape pca sky spectrum: ', rb['pca_sky_spectrum'].shape)

    ### HIER
    rb['sky_subtracted'] = rb['sky_subtracted'] + rb['sky_spectrum'] - rb['pca_sky_spectrum']

    pickle.dump(  ( ww,rb), open(pca_fname,'wb') , protocol=2   )
    print("Wrote ", pca_fname)


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

# find names of all IFUs
ff_1 = glob.glob("/data/hetdex/u/mxhf/rebin/2018????v???/exp0?/multi_???_???_???_??_rebin.pickle")
ifus = []
for f in ff_1:
    ifu = f.split("/")[8][10:13]
    ifus.append(ifu)
ifus = np.unique(ifus)


for ifuslot in ifus:
    ff_022_LL = glob.glob("/data/hetdex/u/mxhf/rebin/2018????v???/exp0?/multi_???_{}_???_??_rebin.pickle".format(ifuslot))

    skys = {}
    ww,skys[("022","LL")],sff = load_skys(ff_022_LL,which="sky_spectrum")

    fiber = 50  # only use one fiber per amplifier, since the sky models are the same
    f1 = np.array([x[fiber] for x in skys[("022","LL")].values()])

    f1[np.isnan(f1)]=1 # get rid of nans
    k1 = np.array([x for x in skys[("022","LL")].keys()]) # get list of keys

    meanmean = np.nanmean(f1[:,450:750], axis=1)
    meanmin=50
    meanmax=300

    ii = (meanmean>=meanmin)&(meanmean<=meanmax)

    #f1[ii].shape

    fcut = f1[ii] # only take spectra where the mean is somewhere nice
    kcut= k1[ii]  # adjust keys

    # get indices and keys for cosmos shots
    cosmosind = []
    cosmoskeys=[]
    for i in range(len(kcut)):
        if kcut[i][0] in cosmosshots:
            cosmosind.append(i)
            cosmoskeys.append(kcut[i])


    #a,b,c,x = 10, 100, 20, np.arange(1010)

    #for i in cosmoskeys:
    #    fcut[i] += gauss(a,b,c,x)

    STDDIV = False # divide by standard deviation or not

    meanf = np.nanmean(fcut, axis=0)
    if STDDIV:
        stdf = np.nanstd(fcut, axis=0)
    else:
        stdf = 1
    fmid = (fcut - meanf)/stdf # normalize spectra

    print('shape fmid: ',fmid.shape)

    fmid[~np.isfinite(fmid)] = 0

    # PCA \(^-^)/
    pca=None
    ncomp = 28
    pca = PCA(n_components=ncomp)

    pca.fit(fmid)

    tA = pca.transform(fmid)
    tA.shape
    # undo normalization
    new = pca.inverse_transform(tA)*stdf+meanf

    # how well did we do?
    print('Cosmoskeys rel res: ',np.nanstd((fcut[cosmosind]-new[cosmosind])/fcut[cosmosind], axis=1).mean())

    # save new sky spectra of cosmos shots in pickle files
    for i in range(len(cosmoskeys)):
        pca_sky = new[cosmosind[i]]
        key = cosmoskeys[i]
        k = (key[0],key[1])
        IFU = key[2]
        amp = key[3]
        save_sky(IFU, amp , k, pca_sky)
