@echo off
set GCS_BUCKET=gs://eo-ald-update
set ROOT_PATH=c:\Users\Chris.Williams\Documents\GitHub
set REPO=ald

REM --------------- plant type ---------------
rem gsutil cp %ROOT_PATH%\%REPO%\cement\plant-type\models\vgg16-256-128.zip %GCS_BUCKET%/%REPO%/cement/plant-type/models/vgg16-256-128.zip
rem gsutil cp %ROOT_PATH%\%REPO%\cement\plant-type\data\chips.zip %GCS_BUCKET%/%REPO%/cement/plant-type/data/chips.zip

REM --------------- production type ---------------
rem gsutil cp %ROOT_PATH%\%REPO%\cement\production-type\models\vgg16-256-128.zip %GCS_BUCKET%/%REPO%/cement/production-type/models/vgg16-256-128.zip
rem gsutil cp %ROOT_PATH%\%REPO%\cement\production-type\data\chips.zip %GCS_BUCKET%/%REPO%/cement/production-type/data/chips.zip

REM --------------- capacity ---------------
rem gsutil cp %ROOT_PATH%\%REPO%\cement\capacity\models\vgg16-256-128.zip %GCS_BUCKET%/%REPO%/cement/capacity/models/vgg16-256-128.zip
rem gsutil cp %ROOT_PATH%\%REPO%\cement\capacity\data\chips.zip %GCS_BUCKET%/%REPO%/cement/capacity/data/chips.zip

REM --------------- footprints ---------------
gsutil cp %ROOT_PATH%\%REPO%\preprocess\data\footprints.zip %GCS_BUCKET%/%REPO%/preprocess/data/footprints.zip
