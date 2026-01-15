#!/bin/bash

#rsync -azv ../b7s26-nagl/initial-force-field.offxml .

#rsync -azv ../b7s26-nagl/training-datasets .

#rsync -azv ../b7s26-nagl/msm-force-field.offxml .

#python 2-curate-training-datasets.py download-optimization                          \
#    --core-opt-dataset      "OpenFF Protein PDB 4-mers v4.0"                        \
#    --initial-force-field   "initial-force-field.offxml"                            \
#    --max-opt-conformers    12                                                      \
#    --output-dataset-path   "training-datasets/protein-4-mer-training-dataset.json" \
#    --output-smirks-path    "training-datasets/protein-4-mer-training-smirks.json"  \
#    --verbose

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

