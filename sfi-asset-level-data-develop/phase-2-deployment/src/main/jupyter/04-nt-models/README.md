# NT Models

Scripts include:

1. `Localizer.ipynb`
2. `Cement vs Steel Features.py`
3. `Localization_classification.py`

Output results include:

1. `sfi-asset-level-data/phase-2-deployment/src/main/resources/nt-model/10km_CS_macro.zip`
2. `sfi-asset-level-data/phase-2-deployment/src/main/resources/nt-model/5km_CS_macro.zip`

These notes will be updated as the model workflow is better understood.

## Notes on output results from email exchange (KS+NT)

*KS:* I'm assuming that these features describe your recommended deployment region for the L8 and S2 deep learning models, but wanted to verify that.

*NT:* Yes. Each grid cell has a probability to contain one or more cement/steel assets. The ‘hotspots’ are geometrically linear because assets are very infrastructure and natural resources dependent. The strength of probability for assets to be found within each cell is described by the attribute ‘preds’

*KS:* Can you give a 1-2 sentence description of what the attributes in this file are? Do I need to consider them for the deployment in any way?

*NT:* The only attribute you may be potentially interested is ‘preds’ (predicted probability of occurrence), in case you find the interest region too big and we will need to narrow it down to the highest probabilities only. As it currently calculated in the attached file \[10km grid\], the region of interest for both types of assets occupies slightly less than 20% of the total region (China) and should capture 95% of all assets.

*NT, regarding 10km grids:* These regions contain 95% of all cement and steel assets, the ones we identified and the potential ones.

*NT, regarding 5km grids:* Attached the macro-localisation regions (87% chance that old/new assets will be found there, this is OSM API, manual download increases accuracy up to 95%, probably something to do with the simplified geometry in the manual version). Col ‘preds’ defines the probability strength of occurrence. 87% means that the rest of assets can be just outside the boxes, the accuracy and hotspot intensity can be increased with resolution. Boxes I sent you are 5*5km, I can make them coarser.

## Notes on scripts from email exchange (KS+NT)

*From NT:* Here are two main scripts for S2. All the training chips have already been uploaded to your servers in the summer. Localization script here contains placeholders for random images, which need to be replaced with S2 training material once connected to EarthAI (this has already been published on the GitHub).