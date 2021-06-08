#!/bin/bash

# run script with ./file_watcher_android.sh <dir_to_watch>
MONITORDIR=$1

inotifywait -m -r -e modify,create,delete "${MONITORDIR}" | while read dir action file
do
  echo "${dir}"
  echo "${MONITORDIR}"
  if [ "${dir}" != "${MONITORDIR}" ]; then
      echo "${file}"
  fi
done
