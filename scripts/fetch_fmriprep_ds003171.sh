#!/usr/bin/env bash
set -e
SKIP_RECONALL=false
while [[ "$1" != "" ]]; do
  case $1 in
    --skip-reconall) SKIP_RECONALL=true ;;
  esac
  shift
done

DOCKER_OPTS=()
if [ "$SKIP_RECONALL" = true ]; then
  DOCKER_OPTS+=(--fs-no-reconall)
elif [ -z "$FS_LICENSE" ]; then
  echo "Error: set FS_LICENSE or pass --skip-reconall"; exit 1
else
  DOCKER_OPTS+=(--fs-license-file /opt/freesurfer/license.txt)
fi

docker run --rm \
  -v "$(pwd)"/data/ds003171:/data:ro \
  -v "$(pwd)"/data/ds003171/derivatives:/out:rw \
  ${FS_LICENSE:+-v "$FS_LICENSE":/opt/freesurfer/license.txt:ro} \
  nipreps/fmriprep:latest \
    /data /out participant --participant-label all \
    "${DOCKER_OPTS[@]}"
