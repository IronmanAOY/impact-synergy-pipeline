#!/usr/bin/env bash
set -e
if [ -z "$FS_LICENSE" ]; then echo "Error: set FS_LICENSE"; exit 1; fi

docker run --rm \
  -v "$(pwd)"/data/scratch/melbourne:/data:ro \
  -v "$(pwd)"/data/scratch/melbourne/derivatives:/out:rw \
  -v "$FS_LICENSE":/opt/freesurfer/license.txt:ro \
  nipreps/fmriprep:latest \
    /data /out participant --participant-label all \
    --fs-license-file /opt/freesurfer/license.txt
