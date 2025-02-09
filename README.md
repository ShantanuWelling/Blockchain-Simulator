# Blockchain-Simulator
### CS765: Introduction to Blockchain, Cryptocurrencies & Smart Contracts [Spring 2025, IIT Bombay]

#### Ameya Deshmukh, Mridul Agarwal, Shantanu Welling
#### 210050011, 210050100, 210010076 

Running instructions: 

`python3 simulator.py -num_peers <val> -frac_slow <val> -frac_low_cpu <val> -I <val> -interarrival_time <val>` 

`<val>` is the value of the parameter. `num_peers` is the number of peers in the network, `frac_slow` is the fraction of slow peers, `frac_low_cpu` is the fraction of peers with low CPU, `I` is average interarrival time between blocks, and `interarrival_time` is the interarrival time between transactions.