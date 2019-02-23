for i in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
do
    name=ciena_HyperOpt_${i}
    echo "bash launch_BDT.sh |& tee $PWD/submit/$name.LOG" \
	| qsub -v "NAME=$name,DATA_PREFIX=$data_prefix,TYPE=$type" \
	-N $name \
	-d $PWD \
	-l nice=0 \
	-j oe \
	-o $PWD/submit 
    sleep 5s
done
