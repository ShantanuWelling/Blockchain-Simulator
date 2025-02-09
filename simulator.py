import random
import argparse
import uuid
import os
from blockchain_lib import *
from typing import Set, Union, Dict, List
import networkx as nx
from matplotlib import pyplot as plt

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
        self.mem_pool: List[Transaction] = []
        self.interarrival_time = interarrival_time
        self.neighbours: List[Peer] = []
        self.transactions_seen: Set[uuid.UUID] = set() ## add tx when generated, received by peer, (or added to blockchain: why?? AD)
        self.blocks_seen: Set[uuid.UUID] = set() ## add block when received by peer or added to blockchain
        self.blockchain_tree = BlockchainTree()
        self.block_being_mined = None
        self.blocks_mined: int = 0

    def generate_transaction(self, timestamp, peers: List["Peer"]) -> Event:
        delay = random.expovariate(1.0 / self.interarrival_time)
        receiver = random.choice([peer for peer in peers if peer.peer_id != self.peer_id])
        
        if self.balance == 0:
            amount = 0
            receiver = self
        else:    
            amount = random.randint(1, self.balance)
        tx_id = uuid.uuid4()
        transaction = Transaction(tx_id, self.peer_id, receiver.peer_id, amount, timestamp + delay)
        event = Event(timestamp + delay, EventType.GENERATE_TRANSACTION, transaction, self.peer_id, None)
        self.transactions_seen.add(tx_id)
        if transaction.amount != 0:
            self.mem_pool.append(transaction)
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
    
    def mine(self, block: Block, timestamp):
        self.blockchain_tree.add(block, timestamp) ## balance map updated
        self.block_being_mined = None
        self.blocks_seen.add(block.block_id)
        ## remove txs from mempool
        self.mem_pool = list(set(self.mem_pool).difference(set(block.transactions)))
        self.blocks_mined = self.blocks_mined + 1

    def choose_transactions(self) -> List[Transaction]:
        transactions: List[Transaction] = []
        balance_map = self.blockchain_tree.longest_chain_leaf.balance_map.copy()
        for tx in sorted(self.mem_pool):
            if len(transactions) == 999:
                break
            if balance_map[tx.sender] >= tx.amount:
                transactions.append(tx)
                balance_map[tx.sender] -= tx.amount
                balance_map[tx.receiver] += tx.amount
        self.balance = balance_map[self.peer_id]
        ## acc to longest chain + block being mined
        return transactions

    def receive_transaction(self, transaction: Transaction) -> bool:
        if transaction.tx_id in self.transactions_seen:
            return False
        self.mem_pool.append(transaction)
        self.transactions_seen.add(transaction.tx_id)
        return True
    
    def receive_block(self, block: Block, timestamp) -> int:
        ## validate transactions
        ## append in own tree
        ## check if already mining and if need to change the block being mined on
        ## -1: already seen block, no forwarding
        # 1: need to re-start mining since longest chain switches, 0: keep mining current block, forward in both
        if block.block_id in self.blocks_seen:
            return -1
        # print("here")
        self.blockchain_tree.add(block, timestamp)
        self.blocks_seen.add(block.block_id)

        longest_chain_leaf = self.blockchain_tree.longest_chain_leaf
        if longest_chain_leaf.block.block_id == self.block_being_mined.parent_block_id:
            return 0
        
        ## updates balance and mem_pool of the peer due to chain switch
        self.balance = longest_chain_leaf.balance_map[self.peer_id]
        old_parent_node = self.blockchain_tree.nodes[self.block_being_mined.parent_block_id]
        old_branch_tx_set, new_branch_tx_set = self.blockchain_tree.lca_branch_txs(old_parent_node)
        mempool_set = set(self.mem_pool)
        mempool_set = mempool_set.difference(new_branch_tx_set).union(old_branch_tx_set)
        self.mem_pool = list(mempool_set)
        return 1 ## this will trigger a start_mining on the new longest chain
    
    def get_ratio(self) -> float:
        my_blocks_in_longest_chain = self.blockchain_tree.number_of_peer_blocks(self.peer_id)
        return my_blocks_in_longest_chain / self.blocks_mined if self.blocks_mined != 0 else 0, my_blocks_in_longest_chain, self.blocks_mined
    
    def visualize_tree(self, output_dir: str):
        G = nx.DiGraph()
        labels = {}

        def add_node_edges(node: BlockChainNode):
            G.add_node(node.block.block_id, height=node.height)  # Add the height attribute here
            receive_time = int(node.receive_timestamp) if node.receive_timestamp is not None else 0
            mine_time = int(node.block.create_timestamp) if node.block.create_timestamp is not None else 0
            labels[node.block.block_id] = f"Miner {node.miner_id}\n N-Txs : {len(node.block.transactions)}\n Mine time: {mine_time}\n Receive time: {receive_time}"
            if node.parent:
                G.add_edge(node.parent.block.block_id, node.block.block_id)

            for child in node.children:
                add_node_edges(child)

        add_node_edges(self.blockchain_tree.root)
        
        node_colors = []

        longest_chain_nodes = set()
        curr_node = self.blockchain_tree.longest_chain_leaf
        while curr_node:
            longest_chain_nodes.add(curr_node.block.block_id)
            curr_node = curr_node.parent
            
        for node in G.nodes:
            if node in longest_chain_nodes:
                node_colors.append('lightgreen')
            else:
                node_colors.append('skyblue')

        pos = nx.multipartite_layout(G, subset_key="height", align='horizontal')
        
        min_height = min([node[1]['height'] for node in G.nodes(data=True)])
        max_height = max([node[1]['height'] for node in G.nodes(data=True)])
        
        for node, (x, y) in pos.items():
            # Invert the y-axis to place root at the top
            pos[node] = (x, max_height - y + min_height)

        # Draw the graph with square-shaped nodes
        plt.figure(figsize=(6, 14))
        nx.draw(G, pos, with_labels=True, labels=labels, node_size=3500, node_color=node_colors, font_size=5, font_weight="bold", width=2, edge_color="gray", node_shape='s')
        
        plt.title(f"Blockchain Tree of Peer {self.peer_id}")
        # plt.show()
        plt.savefig(f"{output_dir}/tree_peer_{self.peer_id}.png", dpi=300, bbox_inches="tight", pad_inches=0)

    def write_to_file(self, file_name: str):
        def add_node_edges_to_file(node: BlockChainNode, file, indent=""):
            file.write(f"{indent}Block ID: {node.block.block_id}\n")
            file.write(f"{indent}Parent Block ID: {node.parent.block.block_id if node.parent else 'None'}\n")            
            file.write(f"{indent}Miner ID: {node.miner_id}\n")
            file.write(f"{indent}Number of Transactions: {len(node.block.transactions)}\n")
            file.write(f"{indent}Mine Time: {int(node.block.create_timestamp) if node.block.create_timestamp is not None else 0}\n")
            file.write(f"{indent}Receive Time: {int(node.receive_timestamp) if node.receive_timestamp is not None else 0}\n")
            file.write(f"{indent}Height: {node.height}\n")
            file.write(f"{indent}" + "-" * 40 + "\n")
            
            # Recursively write child blocks with increased indentation
            for child in node.children:
                add_node_edges_to_file(child, file, indent + "    ")

        with open(file_name, 'w') as file:
            file.write(f"Blockchain Tree of Peer {self.peer_id}\n")
            file.write("=" * 40 + "\n")
            add_node_edges_to_file(self.blockchain_tree.root, file)



class P2PNetwork:
    def __init__(self, num_peers: int, frac_slow: bool, frac_low_cpu: bool, interarrival_time: float, I: float): ## z0 is frac_slow, z1 is frac_low_cpu
        self.peers: List[Peer] = []
        self.num_peers = num_peers
        self.frac_slow = frac_slow
        self.frac_low_cpu = frac_low_cpu
        self.I = I
        self.low_hashing_power = 1 / (num_peers * (10 - 9 * frac_low_cpu))
        self.high_hashing_power = 10 / (num_peers * (10 - 9 * frac_low_cpu))
        self.interarrival_time = interarrival_time
        self.latencies = {}
        self.continue_simulation = True
        self.create_peers()
        self.connect_peers()
        self.initialize_latencies()
        self.initialize_event_queue()


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
            for neighbor in peer.neighbours:
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
            # print(self.latencies)
            latencies = self.latencies[(peer.peer_id, neighbour.peer_id)]
            rho = latencies["rho"]
            c = latencies["c"]
            d = random.expovariate((12/1024) / c)
            message_size = 1 if isinstance(data, Transaction) else len(data.transactions) ## number of KBs
            receiver_delay = rho + message_size * TX_SIZE / c + d
            event_type = EventType.RECEIVE_TRANSACTION if isinstance(data, Transaction) else EventType.RECEIVE_BLOCK
            event = Event(timestamp + receiver_delay, event_type, data, peer.peer_id, neighbour.peer_id)
            self.event_queue.add_event(event)

    def process_events(self, output_dir: str, stopping_height: int, suppress_output: bool):
        while self.continue_simulation:
            while self.event_queue.queue:
                # input()
                event = self.event_queue.pop_event()
                if event.event_type == EventType.GENERATE_TRANSACTION:
                    self.process_generate_transaction(event)
                elif event.event_type == EventType.RECEIVE_TRANSACTION:
                    self.process_receive_transaction(event)
                elif event.event_type == EventType.END_MINING:
                    self.process_end_mining(event)
                elif event.event_type == EventType.RECEIVE_BLOCK:
                    self.process_receive_block(event)
                    
                # check if all peers have reached stopping_height
                if all(peer.blockchain_tree.longest_chain_leaf.height >= stopping_height for peer in self.peers):
                        self.write_balances(f"{output_dir}/balances.txt")
                        self.write_ratios(f"{output_dir}/ratios.txt")
                        for peer in self.peers:
                            peer.write_to_file(f"{output_dir}/tree_peer_{peer.peer_id}.txt")
                            if not suppress_output:
                                peer.visualize_tree(output_dir = output_dir)
                    
                    print(f"All peers have reached height {stopping_height}. Exiting simulation at time {event.timestamp}...")
                    self.continue_simulation = False
                    break 
                    
    
    def process_generate_transaction(self, event: Event):
        id = event.sender
        timestamp = event.timestamp
        peer = self.peers[id]
        gen_tx_event = peer.generate_transaction(timestamp, self.peers)
        self.event_queue.add_event(gen_tx_event) ## add to event queue to generate next transaction
        if event.data.amount != 0:
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
        if block.block_id != peer.block_being_mined.block_id:
            end_mining_event = peer.start_mining(event.timestamp, hashing_power, self.I)
            self.event_queue.add_event(end_mining_event)
            return
        peer.mine(block, event.timestamp) 
        self.forward_packet(peer, block, event.timestamp, id)
        end_mining_event = peer.start_mining(event.timestamp, hashing_power, self.I)
        self.event_queue.add_event(end_mining_event)

    def process_receive_block(self, event: Event):
        receiver = self.peers[event.receiver]
        block = event.data
        hashing_power = self.low_hashing_power if receiver.is_low_cpu else self.high_hashing_power
        return_code = receiver.receive_block(block, event.timestamp)
        # print(return_code, )
        if return_code != -1:
            self.forward_packet(receiver, block, event.timestamp, event.sender)
        if return_code == 1:
            receiver.start_mining(event.timestamp, hashing_power, self.I)
            
    def write_balances(self, file_name: str):
        for peer in self.peers:
            balance_map = peer.blockchain_tree.longest_chain_leaf.balance_map
            # sort balance map by keys
            sorted_balance_map = dict(sorted(balance_map.items()))
            with open(file_name, 'w+') as file:
                file.write(f"Peer {peer.peer_id} balance map: {sorted_balance_map}\n")
                file.write(f"Sum balance map {sum(sorted_balance_map.values())}\n")
    
    def print_blockchain_tree_height(self):
        for peer in self.peers:
            print(f"Peer {peer.peer_id} has blockchain height {peer.blockchain_tree.longest_chain_leaf.height}")
        
    def write_ratios(self, file_name: str):
        ratio_map = {'slow_low': [], 'slow_high': [], 'fast_low': [], 'fast_high': []}
        for peer in self.peers:
            if peer.is_slow and peer.is_low_cpu:
                ratio_map['slow_low'].append(peer.get_ratio())
            elif peer.is_slow and not peer.is_low_cpu:
                ratio_map['slow_high'].append(peer.get_ratio())
            elif not peer.is_slow and peer.is_low_cpu:
                ratio_map['fast_low'].append(peer.get_ratio())
            else:
                ratio_map['fast_high'].append(peer.get_ratio())
        
        for key in ratio_map:
            ratios = [row[0] for row in ratio_map[key]]
            my_blocks_in_longest_chain = [row[1] for row in ratio_map[key]]
            blocks_mined = [row[2] for row in ratio_map[key]]
            # write to file_name
            with open(file_name, 'w+') as file:
                avg_ratio = sum(ratios) / len(ratios) if len(ratios) != 0 else 0
                avg_blocks_longest_chain = sum(my_blocks_in_longest_chain) / len(my_blocks_in_longest_chain) if len(my_blocks_in_longest_chain) != 0 else 0
                avg_blocks_mined = sum(blocks_mined) / len(blocks_mined) if len(blocks_mined) != 0 else 0
                
                file.write(f"Average ratio for {key}: {avg_ratio}\n")
                file.write(f"Average blocks in longest chain for {key}: {avg_blocks_longest_chain}\n")
                file.write(f"Average blocks mined for {key}: {avg_blocks_mined}\n")
            
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='P2P Network Simulator')
    parser.add_argument('-num_peers', type=int, help='Number of peers in the network', required=True)
    parser.add_argument('-frac_slow', type=float, help='Fraction of slow peers', required=True)
    parser.add_argument('-frac_low_cpu', type=float, help='Fraction of low CPU peers', required=True)
    parser.add_argument('-interarrival_time', type=float, help='Mean interarrival time of transactions', required=True)
    parser.add_argument('-I', type=float, help='Block mining time', required=True)
    parser.add_argument('-stopping_height', type=int, help='Simulation stopping criterion', default=10)
    parser.add_argument('-v', type=bool, help='Verbose, logs, plots', default=True)
    args = parser.parse_args()

    num_peers = args.num_peers
    frac_slow = args.frac_slow
    frac_low_cpu = args.frac_low_cpu
    interarrival_time = args.interarrival_time
    stopping_height = args.stopping_height
    suppress_output = not args.v
    I = args.I
    output_dir = f"results_{num_peers}_{int(100*frac_slow)}_{int(100*frac_low_cpu)}_{int(interarrival_time)}_{int(I)}"
    os.makedirs(output_dir, exist_ok = True)
    
    network = P2PNetwork(num_peers, frac_slow, frac_low_cpu, interarrival_time, I)
    network.process_events(output_dir = output_dir, stopping_height = stopping_height, suppress_output = suppress_output)


        
