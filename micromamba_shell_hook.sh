#!/usr/bin/env bash
# ---------------- FHS micromamba shell ----------------
# Environment variables
export AWS_S3_ENDPOINT=s3.af-south-1.amazonaws.com
export AWS_DEFAULT_REGION=af-south-1
export AWS_NO_SIGN_REQUEST=YES
export PIP_NO_CACHE_DIR=1

ENV_NAME=$(yq -r '.name' "$ENV_YAML")

# Keep record of the env files hash files
ENV_YAML_HASH=$(sha256sum "$ENV_YAML" | awk '{print $1}')
REQS_TXT_HASH=$(sha256sum "$REQS_TXT" | awk '{print $1}')

HASH_FILE="$TMPDIR/${ENV_NAME}_hash.json"
mkdir -p "$(dirname "$HASH_FILE")"

# Auto-create or update environment
if ! micromamba env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Creating micromamba environment '$ENV_NAME'..."
    micromamba create -n "$ENV_NAME" -f "$ENV_YAML" -y
    micromamba run -n "$ENV_NAME" pip install -r "$REQS_TXT"
    
    jq -n \
        --arg ENV_YAML_HASH "$ENV_YAML_HASH" \
        --arg REQS_TXT_HASH "$REQS_TXT_HASH" \
        '{ENV_YAML_HASH: $ENV_YAML_HASH, REQS_TXT_HASH: $REQS_TXT_HASH}' > "$HASH_FILE"
else
    # Update ENV_YAML_HASH if changed
    if [ ! -f "$HASH_FILE" ] || [ "$ENV_YAML_HASH" != "$(jq -r '.ENV_YAML_HASH' "$HASH_FILE")" ]; then
        echo "Updating environment conda dependencies in micromamba environment '$ENV_NAME'..."
        micromamba install -n "$ENV_NAME" -f "$ENV_YAML" -y
        jq --arg new "$ENV_YAML_HASH" '.ENV_YAML_HASH = $new' "$HASH_FILE" | sponge "$HASH_FILE"
    fi

    # Update REQS_TXT_HASH if changed
    if [ ! -f "$HASH_FILE" ] || [ "$REQS_TXT_HASH" != "$(jq -r '.REQS_TXT_HASH' "$HASH_FILE")" ]; then
        echo "Updating pip dependencies in micromamba environment '$ENV_NAME'..."
        micromamba run -n "$ENV_NAME" pip install -r "$REQS_TXT"
        jq --arg new "$REQS_TXT_HASH" '.REQS_TXT_HASH = $new' "$HASH_FILE" | sponge "$HASH_FILE"
    fi
fi

# Activate environment
eval "$(micromamba shell hook --shell bash)"
micromamba activate "$ENV_NAME"

# Start interactive shell
exec bash
