## Full-frame sky subtraction

Clone Greg's Panacea repository and copy ffskysub.py to the directory. The code uses Greg's tools.
Run python ffskysub.py --shot DATEVOBS --exp DITHER, where DATEVOBS is e.g. 20190101v016 and DITHER could be exp01, exp02, or exp03.
It runs in python2 and python3 and takes about 5 to 6 minutes to run. 


## PCA sky subtraction

### people from the LSS group :)

Look at the PCA-skysub.ipynb notebook.

### otherwise - Bachelor thesis version

Use on TACC.

Go to the directory where you want to have the pcaskysubtracted multifits files.

Write a list of the shots you want to use in the program pcacommonifu_rebin.py in the beginning: shots = [...], e.g. ['20180822v008', '20180822v009', '20180822v020', '20180822v021', '20180822v022', '20180822v023'].

Run pcacommonifu_rebin.py
