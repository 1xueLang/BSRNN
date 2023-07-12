run() {
    if which python 2> /dev/null; then
        python sdigit.py --sparsity $1 --num_epochs 200 --gamma $2
    else
        python3 sdigit.py --sparsity $1 --num_epochs 200 --gamma $2
    fi
}

S_ARRAY=(1.0 0.8 0.5 0.3 0.1)

for sparsity in ${S_ARRAY[@]}; do
    run $sparsity $1
done