# TTV Search

## Scripts from 2025 Senior Research project

### Overview

The main script is batch_ttv.py, and the transit fitting uses the package juliet. It can be run with the ```--no_ttv``` flag, and the script will skip the TTV fits, useful for testing. An additional script, check_for_lightcurves.py, will plot all the available lightcurves for the systems in systems.csv. systems.csv shows how to format this file, which is used by both scripts. Script for checking lightcurves only need the system names, so you don't need to fill out the rest of the info if you are just checking for data.

There is an additional, newer script in the GP_fitting folder that can be run with the flags '''--gp_matern''' or '''gp_qp''' to do a GP fit first, then detrend the lightcurve, then run the transit fits. This is the idea at least, I have not tested the fitting of the detrended lightcurves yet, as I was doing a lot of testing with this GP fitting. See the To-do below. The '''juliet''' docs have a section on GP fitting, and I was following this, so in theory, it should be easy to just keep following their tutorials.

'''juliet''' docs can be found [here](https://juliet.readthedocs.io/en/latest/)

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
- lightkurve

All other dependencies should be installed along with these packages.  

```
pip install juliet ldtk pandas brokenaxes lightkurve
```

### To-do

- In the GP_fitting script, support for multiple instruments has been added, and I think in general it is working. For certain systems with lots of stellar variability, the GP part of the model seems to have a flat component and I am not sure how to fix this.

- There are other packages that could be used, ```exoplanet``` being the main one. This might be more robust, could be looked into. Also ```exoplanet``` works with Python 3.12, so much more current.
