run() {
    if which python 2> /dev/null; then
        python sdigit.py --percent $1 --gamma $2
    else
        python3 sdigit.py --percent $1 --gamma $2
    fi
}
# 9:1 4:1 3:1 3:2 1:1 
P_ARRAY=(0.8 0.75 0.6)

for percent in ${P_ARRAY[@]}; do
    Loop=$1
    while [ $Loop -gt 0 ]; do
        run $percent $Loop
        let '--Loop'
    done
done