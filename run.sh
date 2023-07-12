# !/bin/bash

# $1 GPU number
# $2 encoder method
# e.g ./run.sh 0 direct
LR=' --lr 0.2'
NUM_EPOCHS=' --num_epochs 200'
MAXDELAY=' --max_delay 2'
GAMMA=' --gamma 0.95'
DIRPREFIX=' --dir_prefix mnistout'


DEVICE=$1
ENCODER=$2

run() {
    args=' --sparsity '$1' --percent '$2' --encoder '$3' --out_type '$4
	CUDA_VISIBLE_DEVICES=$DEVICE python main.py $args$LR$NUM_EPOCHS$GAMMA$DIRPREFIX
}

# sparsity
S_ARRAY=(0.3 0.2)
# percent
P_ARRAY=(0.5 0.75 0.8)
# encoder
E_ARRAY=('poisson' 'ttfs' 'direct')
# out type
O_ARRAY=('psp' 'spike')

for sparsity in ${S_ARRAY[@]}; do
    for percent in ${P_ARRAY[@]}; do
        for out_type in ${O_ARRAY[*]}; do
            run $sparsity $percent $ENCODER $out_type
        done
    done
done