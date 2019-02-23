#!/bin/bash 
:: Random hyper parameter optimization in batch mode for NN

. /cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase/user/atlasLocalSetup.sh
. $ATLAS_LOCAL_ROOT_BASE/packageSetups/localSetup.sh "root 6.14.04-x86_64-slc6-gcc62-opt"

numlayer=$(( ( RANDOM % 5 )  + 1 ))
numn=$(( ( ( RANDOM %30 )  + 1 ) * 10 ))
epochs=4000
opt=$(( ( RANDOM % 2 )))
dropout=$(python -c "import random;print(random.uniform(0.0, 1.0))")
patience=$(( ( RANDOM % 20 )  + 1 )) 

name=${opt}_${numlayer}_${numn}_$(printf %.5f ${dropout})_${patience}_

echo "==> Running the Neural Network Optimisation"

python2 Train_NN.py \
        --v 2 \
        --epochs $epochs \
        --opt $opt \
        --numn $numn \
        --numlayer $numlayer \
        --dropout $(printf %.5f $dropout) \
        --patience $patience \
        --output $name
exit 0
