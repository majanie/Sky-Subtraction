import glob
import pickle
import numpy as np
import os
from sklearn.decomposition import PCA

"""Applies PCA on mean IFU spectrum (for each IFU) and computes corresponding quasi principal components
for one fiber in each amplifier."""

#def gauss(a,b,c,x):
#    return a*np.exp(-(x-b)**2/c)


def load_skys(ff,which="sky_spectrum"): # loads rebinned sky spectra from pickle files
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
        fiber = 50
        try:
            ww,rb = pickle.load( open(f,'rb'), encoding='iso-8859-1' )
            skys[(shotid,exp,ifu,amp)] = rb[which][fiber]/rb["fiber_to_fiber"][fiber]
            sff.append(f)
        except:
            print("Error loading {}.".format(f))
            pass
    print("starting wl = ", ww[0], "A")
    return ww, skys, sff

def save_sky(IFU, amp , k, pca_sky): # saves new spectra in pickle files #/data/hetdex/u/mxhf/rebin/20170331v006/exp01
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
    pca_fname = os.path.join(h2,"meanpca_" + t)

    ww,rb = pickle.load( open(fname,'rb'), encoding='iso-8859-1' )
    #rb["fiber_to_fiber"] = rb["fiber_to_fiber"][:,:N]
    rb["pca_sky_spectrum"] = rb["sky_spectrum"].copy()
    #b["pca_sky_spectrum"][:,:N][:,ii] = pca_sky * rb["fiber_to_fiber"][:,:N][:,ii] 
    rb["pca_sky_spectrum"] = pca_sky * rb["fiber_to_fiber"]

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
print("ifus: ", ifus)

amps = ["LL","LU","RL","RU"]

for ifuslot in ifus:
    ff_022_LL = glob.glob("/data/hetdex/u/mxhf/rebin/2018????v???/exp0?/multi_???_{}_???_LL_rebin.pickle".format(ifuslot)) #np.concatenate([ff_022_LL1,ff_022_LL2,ff_022_LL3]) #
    ff_022_LU = glob.glob("/data/hetdex/u/mxhf/rebin/2018????v???/exp0?/multi_???_{}_???_LU_rebin.pickle".format(ifuslot))
    ff_022_RL = glob.glob("/data/hetdex/u/mxhf/rebin/2018????v???/exp0?/multi_???_{}_???_RL_rebin.pickle".format(ifuslot))
    ff_022_RU = glob.glob("/data/hetdex/u/mxhf/rebin/2018????v???/exp0?/multi_???_{}_???_RU_rebin.pickle".format(ifuslot))

    skys = {}
    ww,skys[("022","LL")],sff = load_skys(ff_022_LL,which="sky_spectrum")
    ww,skys[("022","LU")],sff = load_skys(ff_022_LU,which="sky_spectrum")
    ww,skys[("022","RL")],sff = load_skys(ff_022_RL,which="sky_spectrum")
    ww,skys[("022","RU")],sff = load_skys(ff_022_RU,which="sky_spectrum")

    # keys for each amplifier
    keysLL = np.array([x for x in skys[("022","LL")].keys()])
    keysLU = np.array([x for x in skys[("022","LU")].keys()])
    keysRL = np.array([x for x in skys[("022","RL")].keys()])
    keysRU = np.array([x for x in skys[("022","RU")].keys()])

    print("key shapes: ", keysLL.shape, keysLU.shape, keysRL.shape, keysRU.shape)
    keykey = {"LL":keysLL,"LU":keysLU,"RL":keysRL,"RU":keysRU}

    for amp in amps:
        ampkey = keykey[amp]
        
        # stack -> mean spectrum of IFU for each exposure
        stack = []
        for key in ampkey:
            i=+1
            tmp = []
            key=tuple(key)
            try:
                tmp.append(skys[("022","LL")][(key[0],key[1],key[2],"LL")])
            except KeyError:
                pass
            try:
                tmp.append(skys[("022","LU")][(key[0],key[1],key[2],"LU")])
            except KeyError:
                pass
            try:
                tmp.append(skys[("022","RL")][(key[0],key[1],key[2],"RL")])
            except KeyError:
                pass
            try:
                tmp.append(skys[("022","RU")][(key[0],key[1],key[2],"RU")])
            except KeyError:
                pass

            stack.append(np.nanmean(tmp, axis=0))

        stack = np.array(stack)
        print("shape stack ",stack.shape)

        f1 = stack

        f1[np.isnan(f1)]=1

        k1 = ampkey

        meanmean = np.nanmean(f1[:,450:750], axis=1)
        meanmin=50
        meanmax=300

        ii = (meanmean>=meanmin)&(meanmean<=meanmax)

        # only use exposures with nice means and adjust keys
        fcut = f1[ii]
        kcut= k1[ii]

        #cosmoskeys = []
        #for i in range(len(kcut)):
        #    if kcut[i][0] in cosmosshots:
        #        cosmoskeys.append(i)

        # get indices and keys for cosmos shots
        cosmosind = []
        cosmoskeys=[]
        for i in range(len(kcut)):
            if kcut[i][0] in cosmosshots:
                cosmosind.append(i)
                cosmoskeys.append(kcut[i])

        # normalize spectra
        meanf = np.nanmean(fcut, axis=0)
        stdf = np.nanstd(fcut, axis=0)
        fmid = (fcut - meanf)/stdf
        
        fmid[np.isnan(fmid)] = 1

        # apply PCA
        pca=None
        ncomp = 20
        pca = PCA(n_components=ncomp)

        pca.fit(fmid)

        tA = pca.transform(fmid)
        tA.shape

        #new = pca.inverse_transform(tA)*stdf+meanf

        # get spectra from one amplifier and normalize them
        f2 = np.array([x for x in skys[("022",amp)].values()])
        f2.shape

        fcut2 = f2[ii]
        fcut2.shape

        meanf2 = np.nanmean(fcut2, axis=0)
        stdf2 = np.nanstd(fcut2,axis=0)

        fmid2 = (fcut2-meanf2)/stdf2

        fmid2[np.isnan(fmid2)] = 1

        fnorm = fmid.copy()
        for i in range(fnorm.shape[0]):
            fnorm /= np.linalg.norm(fnorm)

        fnorm2 = fmid2.copy()
        for i in range(fnorm2.shape[0]):
            fnorm2 /= np.linalg.norm(fnorm2)
 
        # get quasi PCs
        quasis = pca.components_ @ fnorm.T @ fnorm2
        qstd = quasis.copy()
        # turn them into unit vectors
        for i in range(quasis.shape[0]):
            qstd[i] = (qstd[i]-np.nanmean(qstd[i]))/np.nanstd(qstd[i])*np.nanstd(pca.components_[i])+np.nanmean(pca.components_[i])
            quasis[i]/= np.linalg.norm(quasis[i])

        # get new sky spectrum 
        qnew = ( tA @ qstd )*stdf2 + meanf2

        print("rel res cosmos shots: ",np.nanstd((fcut2[cosmosind]-qnew[cosmosind])/fcut[cosmosind], axis=1).mean())

        # save new sky spectra of cosmos shots in pickle files
        for i in range(len(cosmoskeys)):
            pca_sky = qnew[cosmosind[i]]
            key = cosmoskeys[i]
            k = (key[0],key[1])
            IFU = key[2]
            amp = key[3]
            save_sky(IFU, amp , k, pca_sky)


