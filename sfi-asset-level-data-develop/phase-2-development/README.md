# Primary Source Code

This is code used for model building and deployment:

* `src/main/jupyter/v4-data-exploration`:
    * Script to explore contents of v4 datasets.
    * Script to create v4.1, which handels duplicated names for assets.
* `src/main/jupyter/S2-macroloc-model`:
    * Macro-localization model that uses Sentinel-2 RGB images to classify chips as cement, steel, or landcover.
    * To be used in deployment phase.
* `src/main/jupyter/tir-macroloc-model`:
    * Macro-localization model that uses Landsat-8 thermal images to classify chips as cement, steel, or landcover.
    * To be used in deployment phase.
* `phase-2-Sentinel`:
    * ?
    
# Primary Resources

* `src/main/resources`:
    * v3, v3.1, v4, and 4.1 of cement and steel assets
* `src/main/resources/cement_steel_land_geoms`:
    * Polygons centered on cement, steel, and landcover assets
* `src/main/resources/annotations`:
    * Annotations of cement plant proporties.
* `src/main/resources/cement_steel_chip_annotations`:
    * Properties of Sentinel-2 chips for macro-localization models

# Exploratory Source Code

This is ancillary code used to explore the data, but is not critical to the final model build or deployment.

* `src/main/jupyter/ALD_TIR_exploration`:
    * Copy of Maral Bayaraa's original code to explore the Landsat-8 thermal model for macrolocalization.
    * Methodolgy replicated in EarthAI platform here: `src/main/jupyter/tir-macroloc-model`.
* `src/main/jupyter/cement-micro-location-model`:
    * Copy of Steve Reece's original code (Phase 1) to create a micro-localization model for cement plants.
    * Decided not to use micro-localization model methodology in Phase 2
* `src/main/jupyter/phase1-data-exploration`:
    * Exploration of Phase 1 datasets (v3).
    * Puts high-resolution cement plant chips used in micro-localization modeling from Phase 1 on shared bucket in AWS.
* `src/main/jupyter/plant-clustering-exploration`:
    * Exploratory modeling using unsupervised techniques to classify plants.
    * Determined that it did not work well and will not use in final deployment.
* `src/main/jupyter/sample-notebooks`:
    * Notebooks that demonstate how to do some things in EarthAI.
* `src/main/jupyter/steel-micro-location-model`: