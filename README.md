# ObsFlow
A workflow to extract, analyze and visualize data of historical climate.

The operations of this workflow are based on and exploit the xscen, xclim and spirograph libraries.

# Features
Extract, analyze and visualize historical climate data from
- AHCCD
- RDRSv2
- ERA5-Land

# Instructions

1. Create env
``` 
mamba env create -f environment.yml
```

2. Get paths_obs.yml from a colleague and modify it with your paths.

3. Modify config_obs.yml for the data you are interested in .

4. Run workflow
```
python workflow_obs.py
```




