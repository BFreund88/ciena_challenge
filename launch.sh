#!/bin/bash 
:: Random hyper parameter optimization in batch mode for NN

. /cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase/user/atlasLocalSetup.sh
. $ATLAS_LOCAL_ROOT_BASE/packageSetups/localSetup.sh "root 6.14.04-x86_64-slc6-gcc62-opt"

numlayer=$(( ( RANDOM % 5 )  + 1 ))
numn=$(( ( ( RANDOM %30 )  + 1 ) * 10 ))
epochs=4000
opt=$(( ( RANDOM % 2 ) ))
dropout=$(python -c "import random;print(random.randint(0, 60)*0.01)")
patience=$(( ( RANDOM % 20 )  + 1 )) 

name=${opt}_${numlayer}_${numn}_${dropout}_${patience}_

echo $dropout
echo "==> Running the Neural Network Optimisation"

python2 Train_NN.py \
        --v 2 \
        --epochs $epochs \
        --opt $opt \
        --numn $numn \
        --numlayer $numlayer \
        --dropout $dropout \
        --patience $patience \
        --output $name
exit 0
