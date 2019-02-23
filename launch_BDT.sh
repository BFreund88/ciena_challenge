#!/bin/bash 
:: Random hyper parameter optimization in batch mode for BDT

. /cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase/user/atlasLocalSetup.sh
. $ATLAS_LOCAL_ROOT_BASE/packageSetups/localSetup.sh "root 6.14.04-x86_64-slc6-gcc62-opt"

nest=$(( ( ( RANDOM % 10 )  + 1 ) * 100  ))
opt=$(( ( RANDOM % 2 ))) 
depth=$(( ( RANDOM % 5 )  + 1 ))
early=$(( ( RANDOM % 10 )  + 1 ))
lr=$(python -c "import random;print(random.uniform(0.005, 0.02))")

name=${opt}_${nest}_${depth}_${lr}_${early}_

echo "==> Running the BDT Optimisation" 

python2 Train_BDT.py \
    --v 2 \
    --opt $opt \
    --depth $depth \
    --lr $lr \
    --early $early \
    --output $name

exit 0