import random
import argparse
import uuid
from blockchain_lib import *

## global constants
TX_SIZE = 1/1024  # 1 KB (in MB)

random.seed(42)


class Peer:
    def __init__(self, peer_id: int, balance: int, is_slow: bool, is_low_cpu: bool, interarrival_time: float):
        self.peer_id = peer_id
        self.balance = balance
        self.is_slow = is_slow
        self.is_low_cpu = is_low_cpu
        self.blockchain = None
        self.mem_pool: set[Transaction] = set()
        self.interarrival_time = interarrival_time
        self.neighbours: list[Peer] = []

    def generate_transaction(self, network: "P2PNetwork", timestamp):

        delay = random.expovariate(1.0 / self.interarrival_time)
        receiver = random.choice([peer for peer in network.peers if peer.peer_id != self.peer_id])
        amount = random.randint(1, self.balance) # Update where? global amount map SW
        tx_id = uuid.uuid4()
        transaction = Transaction(tx_id, self.peer_id, receiver.peer_id, amount)
        event = Event(timestamp + delay, EventType.GENERATE_TRANSACTION, transaction, self.peer_id, None)
        network.event_queue.add_event(event)

        for neighbour in self.neighbours:
            latencies = network.latencies[(self.peer_id, neighbour.peer_id)]
            rho = latencies["rho"]
            c = latencies["c"]
            d = random.expovariate((12/1024) / c)
            receiver_delay = rho + TX_SIZE / c + d + delay
            event = Event(timestamp + receiver_delay, EventType.RECEIVE_TRANSACTION, transaction, self.peer_id, neighbour.peer_id)
            network.event_queue.add_event(event)

class P2PNetwork:
    def __init__(self, num_peers: int, frac_slow: bool, frac_low_cpu: bool, interarrival_time: float): ## z0 is frac_slow, z1 is frac_low_cpu
        self.peers: list[Peer] = []
        self.num_peers = num_peers
        self.frac_slow = frac_slow
        self.frac_low_cpu = frac_low_cpu
        self.interarrival_time = interarrival_time
        self.initialize_event_queue()
        self.create_peers()
        self.connect_peers()
        self.latencies = {}
        self.continue_simulation = True


    def initialize_event_queue(self):
        self.event_queue = EventQueue()
        for i in range(self.num_peers):
            sender = self.peers[i]
            timestamp = random.expovariate(1.0 / self.interarrival_time)
            
            receiver = random.choice([peer for peer in self.peers if peer.peer_id != sender.peer_id])
            amount = random.randint(1, sender.balance) # Update where? global amount map SW
            tx_id = uuid.uuid4()
            transaction = Transaction(tx_id, sender.peer_id, receiver.peer_id, amount)
            event = Event(timestamp, EventType.GENERATE_TRANSACTION, transaction, sender.peer_id, None)
            self.event_queue.add_event(event)

            for neighbour in self.peers[i].neighbours:
                latencies = network.latencies[(sender.peer_id, neighbour.peer_id)]
                rho = latencies["rho"]
                c = latencies["c"]
                d = random.expovariate((12/1024) / c)
                delay = rho + TX_SIZE / c + d
                event = Event(timestamp + delay, EventType.RECEIVE_TRANSACTION, transaction, sender.peer_id, neighbour.peer_id)
                network.event_queue.add_event(event)

    def create_peers(self):
        for i in range(self.num_peers):
            init_balance = 0
            is_slow = random.random() < self.frac_slow
            is_low_cpu = random.random() < self.frac_low_cpu
            peer = Peer(i, init_balance, is_slow, is_low_cpu, self.interarrival_time)
            self.peers.append(peer)

    def connect_peers(self):
        while True:
            for peer in self.peers:
                peer.neighbours = []
            
            for peer in self.peers:
                num_connections = random.randint(3, 6)
                while len(peer.neighbours) < num_connections:
                    neighbour = random.choice(self.peers)
                    if neighbour != peer and neighbour not in peer.neighbours:
                        peer.neighbours.append(neighbour)
                        neighbour.neighbours.append(peer)
            
            if self.check_connectivity():
                break

    def check_connectivity(self):
        visited = [False] * self.num_peers
        stack = [0]
        visited[0] = True
        count = 1
        while stack:
            peer_id = stack.pop()
            for neighbour in self.peers[peer_id].neighbours:
                if not visited[neighbour.peer_id]:
                    visited[neighbour.peer_id] = True
                    stack.append(neighbour.peer_id)
                    count += 1
        return count == self.num_peers
    
    def initialize_latencies(self):
        for peer in self.peers:
            for neighbor in peer.connections:
                if (peer.peer_id, neighbor.peer_id) not in self.latencies:
                    rho = random.uniform(10, 500)  # Propagation delay in ms
                    link_speed = 100 if not (peer.is_slow or neighbor.is_slow) else 5  # 100 Mbps or 5 Mbps
                    self.latencies[(peer.peer_id, neighbor.peer_id)] = {
                        "rho": rho / 1000,  # Convert to seconds
                        "c": link_speed,
                    }
                    self.latencies[(neighbor.peer_id, peer.peer_id)] = self.latencies[(peer.peer_id, neighbor.peer_id)]

    def process_events(self):
        while self.continue_simulation:
            while self.event_queue.queue:
                event = self.event_queue.pop_event()
                if event.event_type == EventType.GENERATE_TRANSACTION:
                    self.process_generate_transaction(event)
                elif event.event_type == EventType.RECEIVE_TRANSACTION:
                    self.process_receive_transaction(event)
    
    def process_generate_transaction(self, event: Event):
        id = event.sender
        timestamp = event.timestamp
        peer = self.peers[id]
        peer.generate_transaction(self, timestamp)

    def process_receive_transaction(self, event: Event):
        transaction = event.data
        receiver = self.peers[event.receiver]
        if transaction in receiver.mem_pool: # or in blockchain SW
            return
        receiver.mem_pool.append(transaction)
        timestamp = event.timestamp
        for neighbour in receiver.neighbours:
            latencies = self.latencies[(receiver.peer_id, neighbour.peer_id)]
            rho = latencies["rho"]
            c = latencies["c"]
            d = random.expovariate((12/1024) / c)
            delay = rho + TX_SIZE / c + d
            event = Event(timestamp + delay, EventType.RECEIVE_TRANSACTION, transaction, receiver.peer_id, neighbour.peer_id)
            



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='P2P Network Simulator')
    parser.add_argument('num_peers', type=int, help='Number of peers in the network')
    parser.add_argument('frac_slow', type=float, help='Fraction of slow peers')
    parser.add_argument('frac_low_cpu', type=float, help='Fraction of low CPU peers')
    parser.add_argument('interarrival_time', type=float, help='Mean interarrival time of transactions')
    args = parser.parse_args()

    num_peers = args.num_peers
    frac_slow = args.frac_slow
    frac_low_cpu = args.frac_low_cpu
    interarrival_time = args.interarrival_time

    network = P2PNetwork(num_peers, frac_slow, frac_low_cpu)
    network.process_events()


        
