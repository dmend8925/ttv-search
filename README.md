# TTV Search 

## Scripts and example outputs from 2025 Senior Research project

### Overview
The main script is batch_ttv.py, and the transit fitting uses the package juliet. It can be run with the ```--no_ttv``` flag, and the script will skip the TTV fits, useful for debugging or testing. An additional script, check_for_lightcurves.py, will plot all the available lightcurves for the systems in systems.csv. systems.csv shows how to format this file, which is used by both scripts. Script for checking lightcurves only need the system names, so you don't need to fill out the rest of the info if you are just checking for data.

Two very useful websites to retrieve data for planets are the [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=PS&constraint=default_flag=1&constraint=disc_facility+like+%27%25TESS%25%27) and [ExoFOP](https://exofop.ipac.caltech.edu/tess/). You can select which columns to display in the archive, and can find most information needed for juliet there. 

### Requirements 
- Python 3.8.2
  - This was the version that I was able to make everything work with. Could maybe be made to work with later versions
- [Multinest](https://github.com/JohannesBuchner/MultiNest)
  - only needed if ```sampler=multinest```, alternatively you can just use ```dynesty```
- [juliet](https://juliet.readthedocs.io/en/latest/index.html)
- [LDTk](https://github.com/hpparvi/ldtk)
- pandas
- brokenaxes

All other dependencies should be installed along with these packages.  
```
pip install juliet ldtk pandas brokenaxes
```

### To-do
- Could implement RV fits as well, ```juliet``` has RV + Transit fit capabilities.
- Need to get fits to converge while assigning each observation as its own "instrument".
  - ```juliet``` lets you combine data from multiple instruments, and *I think* that each TESS mission needs to have its own instrument parameters, if not every separate observation. This is obvious when looking at data that was taken during different missions. There is some backbone in the script to do this, but the fits were not successful, so I resorted to treating every observation as one. See ```juliet``` docs for more information.
- Add in some simple sinusoidal function fitting capability for the O-C plot. ```scipy``` could be used for this.
- There are other packages that could be used, ```exoplanet``` being the main one. This might be more robust, could be looked into.
