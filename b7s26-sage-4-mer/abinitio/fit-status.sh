#!/bin/bash

if [ -z "$1" ]; then

    JOBID=$(squeue -u ccavende -o '%.7i %.4P %.35j %.4u %.2t %.10M %.10l %.2C %.2D %R' \
        | awk '/b7s26-sage-4-mer-driver/ {++n; if (n == 1) print $1}')
    awk -f fit-status.awk b7s26-sage-4-mer-driver-${JOBID}.out

else

    awk -f fit-status.awk "$1"

fi

