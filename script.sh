#! /bin/bash
stopping_height=20

# frac slow 0.2, 0.5, 0.8
# frac low cpu 0.2, 0.5, 0.8
# I 400, 600
# interarrival time 100, 150, 200
# num peers 50, 75
for frac_slow in 0.2 0.5 0.8; do
    for frac_low_cpu in 0.2 0.5 0.8; do
        for I in 400 600; do
            for interarrival_time in 100 150 200 300; do
                for num_peers in 50 80; do
                    python3 simulator.py -num_peers $num_peers -frac_slow $frac_slow -frac_low_cpu $frac_low_cpu -I $I -interarrival_time $interarrival_time -stopping_height $stopping_height &
                done
            done
        done
    done
done

