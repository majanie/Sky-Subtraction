import numpy as np

def get_sky_model(rebinned):
    sky_model = np.mean( rebinned['sky_spectrum']/rebinned['fiber_to_fiber'] , axis = 0)
    return sky_model

def get_rel_throughput(sky_models, shot, common_sky_models):
    
    rel_throughput = {}
    
    ''' key = (exp, ifuslot, amp) '''
    for key in sky_models[shot].keys():
        rel_throughput[key] = sky_models[shot][key] / common_sky_models[(shot,key[0])]
            
    return rel_throughput

def get_x_throughput(sky_models, trialshot, rel_throughput, shots, rebinned):
    
    xrel_throughput = {}
    
    ''' key = (exp, ifuslot, amp) 
    xxrel_throughput = mean of rel_throughput of the same key in all shots '''
    
    for key in rebinned[trialshot].keys(): # changed sky_models[trialshot].keys() to rebinned[trialshot].keys() so we get xsky spectra for flagged ifus
        da = []
        for shot in shots[np.where(shots != trialshot)]: # so that the trialshot rel_throughput is not used
             try:
                 da.append(rel_throughput[shot][key])
             except KeyError:
                 pass
        print('number of valid rel throughputs for xrel_throughput: ', len(da))
        if len(da)>0:
             xrel_throughput[key] = np.nanmean(da, axis=0)
        else:
             print('xrel_throughput of {} in shot {} could not be computed, because the amplifier does not exist in other shots.')
             try:
                 rt = rel_throughput[trialshot][key]
                 xrel_throughput[key] = rt
             except KeyError:
                 pass

    return xrel_throughput 

def get_xsky_spectrum(rel_throughput, shot, common_sky_models, rebinned):
    
    ''' for xsky and xxsky spectrum 
    xsky_spectrum = rel throughput * common sky model * fiber to fiber '''
    
    xsky_models = {}
    xsky_spectrum = {}
    
    for key in rel_throughput[shot].keys():
        xsky_models[key] = rel_throughput[shot][key] * common_sky_models[(shot,key[0])]
        xsky_spectrum[key] = (  xsky_models[key] * rebinned[shot][key]['fiber_to_fiber'] )
                                                     
    return xsky_spectrum 

def main(shots, ww, rebinned, flagged_ifus={}):
 
    ee, aa = ['exp01', 'exp02', 'exp03'], ['LL', 'LU', 'RL', 'RU']

    ss = {}
    fuslots = []
    
    for shot in shots:
        ss[shot] = np.unique([key[1] for key in list(rebinned[shot].keys())])
    
    for key1 in flagged_ifus.keys():
        try:
            for ifu in flagged_ifus[key1]:
                ss[key1] = np.delete( ss[key1], np.where(ss[key1] == ifu ))
        except KeyError:
                print('flagged_ifus contains unused shots as keys')
                pass

    sky_models = {}
    common_sky_models = {}
    rel_throughput = {}

    ''' common sky models '''
    for shot in shots:
        sky_models[shot] = {}
        for exp in ee:
            etwas=[]
            for ifuslot in ss[shot]:
                for amp in aa:
                    try:
                        sky_models[shot][(exp, ifuslot, amp)] = get_sky_model(rebinned[shot][(exp, ifuslot, amp)])
                        etwas.append(sky_models[shot][(exp, ifuslot,amp)])
                    except KeyError:
                        pass
            common_sky_models[(shot,exp)] = np.nanmean(etwas, axis=0)

    ''' rel throughput and xsky spectrum '''   
    for shot in shots:
        rel_throughput[shot] = get_rel_throughput(sky_models, shot, common_sky_models)

            
    ''' xxrel_throughput and xxsky spectrum
        xxrel throughput = mean of rel througput of one (exp, ifuslot, amp)
        xxsky spectrum   = xxrel_throughput * common sky spectrum * fiber to fiber '''
    xrel_throughput = {}
    xsky_spectrum = {}
    for shot in shots:
        xrel_throughput[shot] = get_x_throughput(sky_models, shot, rel_throughput, shots=shots, rebinned = rebinned)
        xsky_spectrum[shot] = get_xsky_spectrum(xrel_throughput, shot, common_sky_models, rebinned)
        
    ''' polynomial fit 
        poly = 5 degree polynomial fit to (sky - xsky)/xsky  for one fiber
        new xsky spectrum = xsky spectrum * ( 1 + poly)'''
    xsky_sub = {}
    rel_error = {}
    for key1 in xsky_spectrum.keys():
        for key2 in xsky_spectrum[key1].keys():
            re = (rebinned[key1][key2]['sky_spectrum'] - xsky_spectrum[key1][key2])[45,:] / xsky_spectrum[key1][key2][45,:]
            sigma = np.nanstd(re[250:])
            sigma2 = np.nanstd(re[:250])
            kappa = 3.5

            flag = np.isfinite(re[250:]) & (np.abs(re[250:]-np.nanmean(re[250:]))<kappa*sigma)
            flag2 = np.isfinite(re[:250]) & (np.abs(re[:250]-np.nanmean(re[250:]))<kappa*sigma2)

            flag2 = np.append(flag2,flag)
            #flag2 = np.array(flag2, dtype=np.int)
            
            try:

                f = np.polyfit(ww[flag2], re[flag2], 5)

                g = f[5] + ww*f[4] + ww*ww*f[3] + ww*ww*ww*f[2]+ ww*ww*ww*ww*f[1]+ ww*ww*ww*ww*ww*f[0]
                                     
                xsky_spectrum[key1][key2] = xsky_spectrum[key1][key2] + xsky_spectrum[key1][key2] * g
                re = re- g
            except IndexError:
                print('A TypeError or IndexError occurred in polynomial fitting.')
                pass
            xsky_sub[(key1, key2)] = rebinned[key1][key2]['spectrum'] - xsky_spectrum[key1][key2]
            rel_error[(key1, key2)] = re
    #rel_error = np.array(rel_error)
    #print('last error shape: ', rel_error.shape)
    return xsky_spectrum, xsky_sub, rel_error
