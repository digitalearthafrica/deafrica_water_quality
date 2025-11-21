#!/usr/bin/env bash
# ---------------- FHS micromamba shell ----------------
# Environment variables
export AWS_S3_ENDPOINT=s3.af-south-1.amazonaws.com
export AWS_DEFAULT_REGION=af-south-1
export AWS_NO_SIGN_REQUEST=YES
export PIP_NO_CACHE_DIR=1

ENV_NAME="deafrica-water-quality-env"

# Keep record of the env files hash files
PROD_YAML_HASH=$(sha256sum "$PROD_YAML" | awk '{print $1}')
PROD_REQS_HASH=$(sha256sum "$PROD_REQS" | awk '{print $1}')
DEV_YAML_HASH=$(sha256sum "$DEV_YAML" | awk '{print $1}')
DEV_REQS_HASH=$(sha256sum "$DEV_REQS" | awk '{print $1}')

HASH_FILE="$TMPDIR/${ENV_NAME}_hash.json"
mkdir -p "$(dirname "$HASH_FILE")"

# Auto-create or update environment
if ! micromamba env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Creating micromamba environment '$ENV_NAME'..."
    micromamba create -n "$ENV_NAME" -f "$PROD_YAML" -y
    micromamba run -n "$ENV_NAME" pip install -r "$PROD_REQS"
    
    micromamba install -n "$ENV_NAME" -f "$DEV_YAML" -y
    jq -n \
        --arg PROD_YAML_HASH "$PROD_YAML_HASH" \
        --arg PROD_REQS_HASH "$PROD_REQS_HASH" \
        --arg DEV_YAML_HASH "$DEV_YAML_HASH" \
        --arg DEV_REQS_HASH "$DEV_REQS_HASH" \
        '{PROD_YAML_HASH: $PROD_YAML_HASH, PROD_REQS_HASH: $PROD_REQS_HASH, DEV_YAML_HASH: $DEV_YAML_HASH, DEV_REQS_HASH: $DEV_REQS_HASH}' > "$HASH_FILE"
else
    # Update PROD_YAML_HASH if changed
    if [ ! -f "$HASH_FILE" ] || [ "$PROD_YAML_HASH" != "$(jq -r '.PROD_YAML_HASH' "$HASH_FILE")" ]; then
        echo "Updating PROD conda dependencies in micromamba environment '$ENV_NAME'..."
        micromamba install -n "$ENV_NAME" -f "$PROD_YAML" -y
        jq --arg new "$PROD_YAML_HASH" '.PROD_YAML_HASH = $new' "$HASH_FILE" | sponge "$HASH_FILE"
    fi

    # Update PROD_REQS_HASH if changed
    if [ ! -f "$HASH_FILE" ] || [ "$PROD_REQS_HASH" != "$(jq -r '.PROD_REQS_HASH' "$HASH_FILE")" ]; then
        echo "Updating PROD pip dependencies in micromamba environment '$ENV_NAME'..."
        micromamba run -n "$ENV_NAME" pip install -r "$PROD_REQS"
        jq --arg new "$PROD_REQS_HASH" '.PROD_REQS_HASH = $new' "$HASH_FILE" | sponge "$HASH_FILE"
    fi

    # Update DEV_YAML_HASH if changed
    if [ ! -f "$HASH_FILE" ] || [ "$DEV_YAML_HASH" != "$(jq -r '.DEV_YAML_HASH' "$HASH_FILE")" ]; then
        echo "Updating DEV conda dependencies in micromamba environment '$ENV_NAME'..."
        micromamba install -n "$ENV_NAME" -f "$DEV_YAML" -y
        jq --arg new "$DEV_YAML_HASH" '.DEV_YAML_HASH = $new' "$HASH_FILE" | sponge "$HASH_FILE"
    fi

    # Update DEV_REQS_HASH if changed
    if [ ! -f "$HASH_FILE" ] || [ "$DEV_REQS_HASH" != "$(jq -r '.DEV_REQS_HASH' "$HASH_FILE")" ]; then
        echo "Updating DEV pip dependencies in micromamba environment '$ENV_NAME'..."
        micromamba run -n "$ENV_NAME" pip install -r "$DEV_REQS"
        jq --arg new "$DEV_REQS_HASH" '.DEV_REQS_HASH = $new' "$HASH_FILE" | sponge "$HASH_FILE"
    fi
fi

# Activate environment
eval "$(micromamba shell hook --shell bash)"
micromamba activate "$ENV_NAME"

# Start interactive shell
exec bash
