#! /bin/bash
stopping_time=20000

# frac slow 0.2, 0.5, 0.8
# frac low cpu 0.2, 0.5, 0.8
# I 400, 600
# interarrival time 100, 150, 200
# num peers 50, 75

for frac_malicious in 0.1 0.2 0.3 0.4 0.5; do
    for timeout in 10 100 1000; do 
        python3 simulator.py -num_peers 100 -frac_slow 0.5 -frac_low_cpu 0.5 -I 400 -interarrival_time 50 -stopping_time 20000 -frac_malicious $frac_malicious -timeout $timeout > $frac_malicious-$timeout.log 2>&1 &
    done
done


