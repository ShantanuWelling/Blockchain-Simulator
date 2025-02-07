import random
import argparse
import uuid
from blockchain_lib import *
from typing import Set, Union, Dict

## global constants
TX_SIZE = 1/1024  # 1 KB (in MB)
COINBASE_REWARD = 50  # 50 coins

random.seed(42)


class Peer:
    def __init__(self, peer_id: int, balance: int, is_slow: bool, is_low_cpu: bool, interarrival_time: float):
        self.peer_id = peer_id
        self.balance = balance
        self.is_slow = is_slow
        self.is_low_cpu = is_low_cpu
        self.mem_pool: list[Transaction] = []
        self.interarrival_time = interarrival_time
        self.neighbours: list[Peer] = []
        self.transactions_seen: Set[uuid.UUID] = set() ## add tx when generated, received by peer, or added to blockchain
        self.blocks_seen: Set[uuid.UUID] = set() ## add block when received by peer or added to blockchain
        self.blockchain_tree = BlockchainTree()
        self.block_being_mined = None

    def generate_transaction(self, timestamp, peers: list["Peer"]) -> Event:
        delay = random.expovariate(1.0 / self.interarrival_time)
        receiver = random.choice([peer for peer in peers if peer.peer_id != self.peer_id])
        amount = random.randint(1, self.balance) # Update where? global amount map SW
        tx_id = uuid.uuid4()
        transaction = Transaction(tx_id, self.peer_id, receiver.peer_id, amount, timestamp + delay)
        event = Event(timestamp + delay, EventType.GENERATE_TRANSACTION, transaction, self.peer_id, None)
        self.transactions_seen.add(tx_id)
        return event

    def start_mining(self, timestamp, hashing_power: float, interarrival_time: int) -> Event:
        parent_block_id = self.blockchain_tree.longest_chain_leaf.block.block_id
        coinbase_transaction = Transaction(uuid.uuid4(), None, self.peer_id, COINBASE_REWARD, timestamp)
        block_transactions = [coinbase_transaction] + self.choose_transactions()
        block_id = uuid.uuid4()
        mining_time = random.expovariate(hashing_power / interarrival_time)
        block = Block(block_id, block_transactions, parent_block_id, timestamp + mining_time)
        self.block_being_mined = block
        event = Event(timestamp + mining_time, EventType.END_MINING, block, self.peer_id, None)
        return event
    
    def mine(self, block: Block):
        self.blockchain_tree.add(block) ## balance map updated
        self.block_being_mined = None
        self.blocks_seen.add(block.block_id)
        ## remove txs from mempool
        self.mem_pool = [tx for tx in self.mem_pool if tx.tx_id not in block.transactions]


    def choose_transactions(self) -> list[Transaction]:
        ## also updates balances of all peers
        transactions: list[Transaction] = []
        balance_map = self.blockchain_tree.longest_chain_leaf.balance_map.copy()
        for tx in sorted(self.mem_pool):
            if len(transactions) == 999:
                break
            if balance_map[tx.sender] >= tx.amount:
                transactions.append(tx)
                balance_map[tx.sender] -= tx.amount
                balance_map[tx.receiver] += tx.amount  
        return transactions

    def receive_transaction(self, transaction: Transaction) -> bool:
        if transaction.tx_id in self.transactions_seen:
            return False
        self.mem_pool.append(transaction)
        self.transactions_seen.add(transaction.tx_id)
        return True
    
    def receive_block(self, block: Block) -> bool:
        ## validate transactions
        ## append in own tree
        ## check if already mining and if need to change the block being mined on

        if block.block_id in self.blocks_seen:
            return False
        self.blockchain_tree.add(block)
        longest_chain_leaf = self.blockchain_tree.longest_chain_leaf.id
        if longest_chain_leaf == self.block_being_mined.parent_block_id:
            return 
        return True

class P2PNetwork:
    def __init__(self, num_peers: int, frac_slow: bool, frac_low_cpu: bool, interarrival_time: float, I: float): ## z0 is frac_slow, z1 is frac_low_cpu
        self.peers: list[Peer] = []
        self.num_peers = num_peers
        self.frac_slow = frac_slow
        self.frac_low_cpu = frac_low_cpu
        self.I = I
        self.low_hashing_power = 1 / (num_peers * (10 - 9 * frac_low_cpu))
        self.high_hashing_power = 10 / (num_peers * (10 - 9 * frac_low_cpu))
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
            gen_tx_event = sender.generate_transaction(0, self.peers)
            self.event_queue.add_event(gen_tx_event)
            self.forward_packet(sender, gen_tx_event.data, gen_tx_event.timestamp, i)
            hashing_power = self.low_hashing_power if sender.is_low_cpu else self.high_hashing_power
            end_mining_event = sender.start_mining(0, hashing_power, self.I)
            self.event_queue.add_event(end_mining_event)

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
    
    def forward_packet(self, peer: Peer, data: Union[Transaction, Block], timestamp, sender_id: int):
        for neighbour in peer.neighbours:
            if neighbour.peer_id == sender_id:
                continue
            latencies = self.latencies[(peer.peer_id, neighbour.peer_id)]
            rho = latencies["rho"]
            c = latencies["c"]
            d = random.expovariate((12/1024) / c)
            message_size = 1 if isinstance(data, Transaction) else len(data.transactions) ## number of KBs
            receiver_delay = rho + message_size * TX_SIZE / c + d
            event_type = EventType.RECEIVE_TRANSACTION if isinstance(data, Transaction) else EventType.RECEIVE_BLOCK
            event = Event(timestamp + receiver_delay, event_type, data, peer.peer_id, neighbour.peer_id)
            self.event_queue.add_event(event)

    def process_events(self):
        while self.continue_simulation:
            while self.event_queue.queue:
                event = self.event_queue.pop_event()
                if event.event_type == EventType.GENERATE_TRANSACTION:
                    self.process_generate_transaction(event)
                elif event.event_type == EventType.RECEIVE_TRANSACTION:
                    self.process_receive_transaction(event)
                elif event.event_type == EventType.END_MINING:
                    self.process_end_mining(event)
                elif event.event_type == EventType.RECEIVE_BLOCK:
                    self.process_receive_block(event)
    
    def process_generate_transaction(self, event: Event):
        id = event.sender
        timestamp = event.timestamp
        peer = self.peers[id]
        gen_tx_event = peer.generate_transaction(self, timestamp, self.peers)
        self.event_queue.add_event(gen_tx_event) ## add to event queue to generate next transaction
        self.forward_packet(peer, gen_tx_event.data, gen_tx_event.timestamp, id)

    def process_receive_transaction(self, event: Event):
        receiver = self.peers[event.receiver]
        transaction = event.data
        unseen_tx = receiver.receive_transaction(transaction)
        if unseen_tx:
            self.forward_packet(receiver, transaction, event.timestamp, event.sender)

    def process_end_mining(self, event: Event):
        id = event.sender
        peer = self.peers[id]
        block = event.data
        hashing_power = self.low_hashing_power if peer.is_low_cpu else self.high_hashing_power
        if block.id != peer.block_being_mined.id:
            end_mining_event = peer.start_mining(event.timestamp, hashing_power, self.I)
            self.event_queue.add_event(end_mining_event)
            return
        peer.mine(block) 
        self.forward_packet(peer, block, event.timestamp, id)
        end_mining_event = peer.start_mining(event.timestamp, hashing_power, self.I)
        self.event_queue.add_event(end_mining_event)

    def process_receive_block(self, event: Event):
        receiver = self.peers[event.receiver]
        block = event.data
        
        unseen_valid_block = receiver.receive_block(block)
        if unseen_valid_block:
            self.forward_packet(receiver, block, event.timestamp, event.sender)
        

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='P2P Network Simulator')
    parser.add_argument('num_peers', type=int, help='Number of peers in the network', required=True)
    parser.add_argument('frac_slow', type=float, help='Fraction of slow peers', required=True)
    parser.add_argument('frac_low_cpu', type=float, help='Fraction of low CPU peers', required=True)
    parser.add_argument('interarrival_time', type=float, help='Mean interarrival time of transactions', required=True)
    parser.add_argument('I', type=float, help='Block mining time', required=True)
    args = parser.parse_args()

    num_peers = args.num_peers
    frac_slow = args.frac_slow
    frac_low_cpu = args.frac_low_cpu
    interarrival_time = args.interarrival_time
    I = args.I

    network = P2PNetwork(num_peers, frac_slow, frac_low_cpu, interarrival_time, I)
    network.process_events()


        
