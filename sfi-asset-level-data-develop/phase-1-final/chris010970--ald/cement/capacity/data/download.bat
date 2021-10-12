@echo off
set GCS_BUCKET=gs://eo-ald-update
set REPO=satellite-image-ai

gsutil cp %GCS_BUCKET%/%REPO%/cement/capacity/data/chips.zip .
