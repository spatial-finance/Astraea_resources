# Google Earth Pro Cement Annotation Task

Instructions here: https://astraea.app.box.com/file/691897961557

### Steps to setup environment (only need to do this one time) 

1. Launch standard size EarthAI Notebooks instance
2. Clone the sfi-asset-level-data repo https://github.com/s22s/sfi-asset-level-data
3. Launch a terminal and switch to the annotations branch `git checkout annotations`

### Steps to run feedback code

1. Launch standard size EarthAI Notebooks instance
2. In box directory https://astraea.app.box.com/folder/134057087450, download the following files/folders:
    - "Astraea to Review" folder
    - "Completed.tar.gz" file
    - "Input Files/Cement Plant Annotation - Group 3.xlsx" file https://astraea.app.box.com/file/790789672957?s=kvyapziit3epr54e3t4dsr723qrh2alk
3. Upload these files to the directory "~/sfi-asset-level-data/phase-2-deployment/src/main/resources/cement-annotations/Group 3"
4. In terminal, go to above directory `cd "sfi-asset-level-data/phase-2-deployment/src/main/resources/cement-annotations/Group 3"`
5. In terminal, unzip "Astraea to Review.zip" `unzip "Astraea to Review.zip"`
6. In terminal, unzip "Completed.tar.gz" `tar -xf Completed.tar.gz`
7. Open "qa-annotations.ipynb" and run all cells
8. Once notebook is complete (should take about 5 min), download the following files:
    - "CloudFactory to Review.tar.gz"
    - "Completed.tar.gz"
    - "Cement Plant Annotation - Group 3.xlsx"
9. In box directory https://astraea.app.box.com/folder/134057087450, clear out the following folders:
    - "Astraea to Review" 
    - "CloudFactory to Review"
10. In box directory https://astraea.app.box.com/folder/134057087450, upload the following files/folders:
    - unzip "CloudFactory to Review.tar.gz" and upload contents to "CloudFactory to Review" folder in Box
    - Completed.tar.gz <- don't need to unzip this
    - "Cement Plant Annotation - Group 3.xlsx" to the "Input Files" directory (replacing the current file)


### Steps to run output code

1. Launch standard size EarthAI Notebooks instance
2. In box directory https://astraea.app.box.com/folder/134057087450, download the following files:
    - "Completed.tar.gz"
    - "Input Files/Cement Plant Annotation - Group 3.xlsx" https://astraea.app.box.com/file/790789672957?s=kvyapziit3epr54e3t4dsr723qrh2alk
3. Upload these files to the directory "~/sfi-asset-level-data/phase-2-deployment/src/main/resources/cement-annotations/Group 3"
4. In terminal, go to above directory `cd "sfi-asset-level-data/phase-2-deployment/src/main/resources/cement-annotations/Group 3"`
5. In terminal, unzip "Completed.tar.gz" `tar -xf Completed.tar.gz`
6. Open "prepare-output.ipynb" and run all cells
7. Once notebook is complete, add/commit the following files to the Github repo (`git add <file>` then `git commit <file> -m <message>`)
    - "Completed.tar.gz"
    - "Cement Plant Annotation - Group 3.xlsx"
    - "output/raw_annotations_set3.geojson"
    - "output/aggregated_annotations_set3.csv"
8. Download the following files and upload to box directory https://astraea.app.box.com/folder/134057087450
    - "Completed.tar.gz"
    - "Cement Plant Annotation - Group 3.xlsx" to the "Input Files" directory
9. Download the following files and send to Cristian at SFI 
    - "output/raw_annotations_set3.geojson"
    - "output/aggregated_annotations_set3.csv"