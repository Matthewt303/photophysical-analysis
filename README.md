## Overview

This repository contains a series of scripts to analyse photoswitching from SMLM image sequences. 

## Installation

```
git clone https://github.com/Matthewt303/photophysical-analysis.git
cd photophysical-analysis
```

## Usage

```analyse.py``` is used to analyse photoswitching from an image sequence. Under the ```main``` function, replace the hyperparameters with your own. E.g.

```python
## Hyperparameters ##

file_name = "your/file/here/image_file.tif"
out = "your/output_folder/here"
exposure time = 0.03 # In seconds.
adc = 0.59 # Analog-digital conversion rate.
```

The ```main``` function of ```ensemble_analysis.py``` is used to generate plots for each photoswitching parameter.
