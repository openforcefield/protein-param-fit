#!/bin/bash

#rsync -azv ../b7s26-sage-pairwise/initial-force-field.offxml .

#rsync -azv ../b7s26-4-mer/training-datasets .

#rsync -azv ../b7s26-sage-pairwise/msm-force-field.offxml .

python 4-create-forcebalance-inputs.py                                                  \
    --tag                       "abinitio/forcebalance"                                 \
    --force-field-path          "msm-force-field.offxml"                                \
    --abinitio-dataset-path     "training-datasets/protein-4-mer-training-dataset.json" \
    --optimization-dataset-path "training-datasets/optimization-training-dataset.json"  \
    --torsiondrive-dataset-path "training-datasets/torsiondrive-training-dataset.json"  \
    --valence-smirks-path       "training-datasets/optimization-training-smirks.json"   \
    --torsion-smirks-path       "training-datasets/torsiondrive-training-smirks.json"   \
    --smarts-to-exclude         "../smarts-to-exclude.dat"                              \
    --smiles-to-exclude         "../smiles-to-exclude.dat"                              \
    --protein-record-ids-path   "../protein-record-ids.dat"                             \
    --abinitio-weight           0.4                                                    \
    --opt-geo-weight            0.005                                                   \
    --port                      55125                                                   \
    --torsiondrive-weight       0.01                                                    \
    --torsiondrive-target-type  "AbInitio"                                              \
    --verbose

sed -i 's/AbInitio_SMIRNOFF/AbInitioPairwise_SMIRNOFF/g' abinitio/forcebalance/optimize.in

