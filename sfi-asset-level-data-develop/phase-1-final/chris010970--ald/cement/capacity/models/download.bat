@echo off
set GCS_BUCKET=gs://eo-ald-update
set REPO=satellite-image-ai

gsutil cp %GCS_BUCKET%/%REPO%/cement/capacity/models/vgg16-256-128.zip .
