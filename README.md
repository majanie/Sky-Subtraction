## PCA sky subtraction

### people from the LSS group :)

Look at the PCA-skysub.ipynb notebook.

### otherwise - Bachelor thesis version

Use on TACC.

Go to the directory where you want to have the pcaskysubtracted multifits files.

Write a list of the shots you want to use in the program pcacommonifu_rebin.py in the beginning: shots = [...], e.g. ['20180822v008', '20180822v009', '20180822v020', '20180822v021', '20180822v022', '20180822v023'].

Run pcacommonifu_rebin.py


## cross sky subtraction

Use on TACC.

Go to the directory where you want to have the xskysubtracted multifits files.

Write a list of the shots you want to use in the program xskysub_new.py in main(): shots = [...], e.g. ['20180822v008', '20180822v009', '20180822v020', '20180822v021', '20180822v022', '20180822v023'].

Write a list of IFUs that don't have a nice sky (e.g. because of a star) in flagged_ifus (a dictionary) with the shot as the key and a list of the bad IFUs in this shot as the value.

Run xskysub_new.py
