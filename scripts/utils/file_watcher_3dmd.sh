#!/bin/bash

# run script with ./file_watcher_android.sh <dir_to_watch> <destination_dir>
MONITOR_DIR=$1
DEST_DIR=$2

inotifywait -m "${MONITOR_DIR}" -e create,moved_to |
  while read dir action filename
    do
      echo "${dir}${filename}"
      extension="${filename##*.}"
      echo "${extension}"

      if [ "${extension}" == "obj" ]; then
        binvox ${dir}${filename}
      fi

      if [ "${extension}" == "binvox" ]; then
        mv ${dir}${filename} ${DEST_DIR}
        python ~/Desktop/Git/3d_cranio_detection/scripts/preprocess/voxel_to_input.py "${filename}"
        name="${filename%.*}"
        echo "${name}"
        python ~/Desktop/Git/3d_cranio_detection/scripts/models/model_projections_2d.py --mode predict --source 3dmd --filename "${name}"
      fi

    done
