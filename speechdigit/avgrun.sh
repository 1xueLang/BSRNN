loop=$1

run() {
    if which python 2> /dev/null; then
        python sdigit.py
    else
        python3 sdigit.py
    fi
}

while [ $loop -gt 0 ]; do
    run    
    echo $loop >> avgrun.ret
    tail -n 1 ../sdigitout/eiDRSNN-T100-LR0.05-G0.97-S0.5-P0.5-D0-abs-spike.log >> avgrun.ret
    let '--loop'
done