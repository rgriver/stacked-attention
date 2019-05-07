#!/bin/bash

NOW=$(date +"%Y%m%d_%H%M%S")
JOB_ID="clevr_$NOW"
BUCKET="gs://rrdata"
GCS_PATH="${BUCKET}/${JOB_ID}"
TRAINER_PACKAGE_PATH="./trainer"
MAIN_TRAINER_MODULE="trainer.train"
REGION="us-central1"
CLOUD_CONFIG="trainer/config.yaml"


# Training
gcloud ai-platform jobs submit training "$JOB_ID" \
    --staging-bucket "$BUCKET" \
    --job-dir "$GCS_PATH"  \
    --package-path "$TRAINER_PACKAGE_PATH" \
    --module-name "$MAIN_TRAINER_MODULE" \
    --region "$REGION" \
    --config "$CLOUD_CONFIG" \
    -- \
    --output_path "${GCS_PATH}"