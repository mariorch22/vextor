#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data"
URL="http://corpus-texmex.irisa.fr/sift.tar.gz"
TARBALL="$DATA_DIR/sift.tar.gz"

mkdir -p "$DATA_DIR"
trap 'rm -f "$TARBALL"' ERR

# Check if already downloaded and extracted.
if [ -f "$DATA_DIR/sift_base.fvecs" ] && \
   [ -f "$DATA_DIR/sift_query.fvecs" ] && \
   [ -f "$DATA_DIR/sift_groundtruth.ivecs" ]; then
    echo "SIFT1M data already present in $DATA_DIR"
    exit 0
fi

echo "Downloading SIFT1M dataset..."
wget -q --show-progress -O "$TARBALL" "$URL"

echo "Extracting..."
tar -xzf "$TARBALL" -C "$DATA_DIR" --strip-components=1

# Verify files exist.
for f in sift_base.fvecs sift_query.fvecs sift_groundtruth.ivecs sift_learn.fvecs; do
    if [ ! -f "$DATA_DIR/$f" ]; then
        echo "ERROR: expected file $DATA_DIR/$f not found after extraction"
        exit 1
    fi
done

# Verify file sizes (approximate).
BASE_SIZE=$(stat -c%s "$DATA_DIR/sift_base.fvecs" 2>/dev/null || stat -f%z "$DATA_DIR/sift_base.fvecs")
if [ "$BASE_SIZE" -lt 500000000 ]; then
    echo "ERROR: sift_base.fvecs seems too small ($BASE_SIZE bytes)"
    exit 1
fi

rm -f "$TARBALL"
echo "SIFT1M data ready in $DATA_DIR"
