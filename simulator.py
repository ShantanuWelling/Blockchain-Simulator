import random
import argparse
import time
import uuid
import os
from blockchain_lib import *
from typing import Set, Union, Dict, List
import networkx as nx
from matplotlib import pyplot as plt

## global constants 
TX_SIZE = 1/1000  # 1 KB (in MB)
COINBASE_REWARD = 50  # 50 coins
random.seed(42)
DEBUG = 0

class Peer:
    def __init__(self, peer_id: int, balance: int, is_slow: bool, is_low_cpu: bool, interarrival_time: float, timeout: int):
        self.peer_id = peer_id
        self.balance = balance ## maintained according to the longest chain + block being mined
        self.is_slow = is_slow
        self.is_low_cpu = is_low_cpu
        self.mem_pool: List[Transaction] = []
        self.interarrival_time = interarrival_time
        self.neighbours: List[Peer] = []
        ## for loopless forwarding
        self.transactions_seen: Set[uuid.UUID] = set()
        # self.blocks_seen: Set[uuid.UUID] = set()
        self.blocks_seen: Dict[int, Block] = defaultdict(list)
        self.blockchain_tree = BlockchainTree()
        ## block being mined currently
        self.block_being_mined = None
        self.blocks_mined: int = 0
        self.malicious = False
        self.timeout = timeout
        self.hashes_seen: Set[int] = set()
        self.pending_blocks: Dict[int, list[int]] = defaultdict(list)

    def generate_transaction(self, timestamp, peers: List["Peer"]) -> Event:
        ## Generates a transaction and returns an event to process the same after an exponential delay
        delay = random.expovariate(1.0 / self.interarrival_time)
        receiver = random.choice([peer for peer in peers if peer.peer_id != self.peer_id])
        
        if self.balance == 0:
            ## dummy 0 amount transaction if no balance
            amount = 0
            receiver = self
        else:    
            amount = random.randint(1, self.balance)
        tx_id = uuid.uuid4() ## unique id per transaction
        transaction = Transaction(tx_id, self.peer_id, receiver.peer_id, amount, timestamp + delay)
        event = Event(timestamp + delay, EventType.GENERATE_TRANSACTION, transaction, self.peer_id, None)
        self.transactions_seen.add(tx_id)
        if transaction.amount != 0:
            ## add non-dummy txs to the mempool
            self.mem_pool.append(transaction)
        return event

    def start_mining(self, timestamp, hashing_power: float, interarrival_time: int) -> Event:
        ## Collect transactions from the mempool, and return an event to end mining after appropriate delay
        parent_block_id = self.blockchain_tree.longest_chain_leaf.block.block_id
        coinbase_transaction = Transaction(uuid.uuid4(), None, self.peer_id, COINBASE_REWARD, timestamp)
        block_transactions = [coinbase_transaction] + self.choose_transactions()
        block_id = uuid.uuid4()
        mining_time = random.expovariate(hashing_power / interarrival_time)
        block = Block(block_id, block_transactions, parent_block_id, timestamp + mining_time)
        self.block_being_mined = block
        event = Event(timestamp + mining_time, EventType.END_MINING, block, self.peer_id, None, self.malicious)
        return event
    
    def mine(self, block: Block, timestamp):
        self.blockchain_tree.add(block, timestamp)
        self.block_being_mined = None
        # self.blocks_seen.add(block.block_id)
        self.blocks_seen[block.hash] = block
        ## remove block txs from mempool
        self.mem_pool = list(set(self.mem_pool).difference(set(block.transactions)))
        self.blocks_mined = self.blocks_mined + 1

    def choose_transactions(self) -> List[Transaction]:
        transactions: List[Transaction] = []
        balance_map = self.blockchain_tree.longest_chain_leaf.balance_map.copy()
        ## greedily pick as many transactions as possible from the mempool
        for tx in sorted(self.mem_pool):
            if len(transactions) == 999:
                break
            if balance_map[tx.sender] >= tx.amount:
                ## only pick consistent transactions
                transactions.append(tx)
                balance_map[tx.sender] -= tx.amount
                balance_map[tx.receiver] += tx.amount
        ## update balance of the peer 
        self.balance = balance_map[self.peer_id]
        return transactions

    def receive_transaction(self, transaction: Transaction) -> bool:
        if transaction.tx_id in self.transactions_seen:
            return False
        self.mem_pool.append(transaction)
        self.transactions_seen.add(transaction.tx_id)
        return True
    
    def receive_block(self, block: Block, timestamp, sender_id: int) -> int:

        ## check if before timeout
        if not self.pending_blocks[block.hash] or self.pending_blocks[block.hash][0] != sender_id:
            return -1
        if block.hash in self.blocks_seen:
            return -1
        self.pending_blocks.pop(block.hash)
        self.blocks_seen[block.hash] = block

        ## -1: already seen block, no forwarding
        ## 1: longest chain switches terminate and re-start mining, forward block
        ## 0: keep mining current block, forward block

        self.blockchain_tree.add(block, timestamp)
        longest_chain_leaf = self.blockchain_tree.longest_chain_leaf
        if longest_chain_leaf.block.block_id == self.block_being_mined.parent_block_id:
            ## longest chain maintained
            return 0

        ## longest chain switches
        ## update self balance
        self.balance = longest_chain_leaf.balance_map[self.peer_id]
        old_parent_node = self.blockchain_tree.nodes[self.block_being_mined.parent_block_id]
        ## update mempool
        ## add the tx set in the old branch upto the LCA
        ## remove those in the new branch from LCA onwards
        old_branch_tx_set, new_branch_tx_set = self.blockchain_tree.lca_branch_txs(old_parent_node)
        mempool_set = set(self.mem_pool)
        mempool_set = mempool_set.union(old_branch_tx_set).difference(new_branch_tx_set)
        self.mem_pool = list(mempool_set)
        return 1
    
    def receive_hash(self, hash: int, sender: int, timestamp: int) -> List[Event]:
        if hash not in self.hashes_seen:
            self.hashes_seen.add(hash)
            self.pending_blocks[hash].append(sender)
            ## Get Request Event to the hash sender + Timeout Event
            request = Event(timestamp, EventType.SEND_REQUEST, hash, self.peer_id, sender)
            timeout = Event(timestamp + self.timeout, EventType.TIMEOUT, hash, self.peer_id, None)
            return (request, timeout)
        else:
            if self.pending_blocks[hash]:
                ## if a timeout is already ongoing, add the hash to the queue for sending request after
                ## current timeout expires
                self.pending_blocks[hash].append(sender)
                return ()
            else:
                self.pending_blocks[hash].append(sender)
                ## Get Request Event to the hash sender + Timeout Event
                request = Event(timestamp, EventType.SEND_REQUEST, hash, self.peer_id, sender)
                timeout = Event(timestamp + self.timeout, EventType.TIMEOUT, hash, self.peer_id, None)
                return (request, timeout)

    def receive_request(self, hash: int): # return block
        return self.blocks_seen[hash]
        
    def process_timeout(self, hash: int, timestamp: int):
        if not self.pending_blocks[hash]:
            return ()
        self.pending_blocks[hash].pop(0) # remove timeout for that hash
        
        # if some other neighbour has also sent the same hash, then send request to that node and start timeout
        if self.pending_blocks[hash]:
            provider = self.pending_blocks[hash][0]
            request = Event(timestamp, EventType.SEND_REQUEST, hash, self.peer_id, provider)
            timeout = Event(timestamp + self.timeout, EventType.TIMEOUT, hash, self.peer_id, None)
            return (request, timeout)
        return ()

    def get_ratio(self) -> float:
        ## Ratio of the number of peer's blocks in the chain to total mined
        my_blocks_in_longest_chain = self.blockchain_tree.number_of_peer_blocks(self.peer_id)
        return my_blocks_in_longest_chain / self.blocks_mined if self.blocks_mined != 0 else 0, my_blocks_in_longest_chain, self.blocks_mined
    
    def visualize_tree(self, output_dir: str, ringleader_id: int):
        ## Visualization code
        G = nx.DiGraph()
        labels = {}

        malicious_nodes = set()
        def add_node_edges(node: BlockChainNode):
            if node.miner_id == ringleader_id:
                malicious_nodes.add(node.block.block_id)
            G.add_node(node.block.block_id, height=node.height)
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
            if node in malicious_nodes:
                node_colors.append('red')
            elif node in longest_chain_nodes:
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
        plt.figure(figsize=(10, 25))
        nx.draw(G, pos, with_labels=True, labels=labels, node_size=3500, node_color=node_colors, font_size=5, font_weight="bold", width=2, edge_color="gray", node_shape='s')
        
        plt.title(f"Blockchain Tree of Peer {self.peer_id}")
        # plt.show()
        plt.savefig(f"{output_dir}/tree_peer_{self.peer_id}.png", dpi=300, bbox_inches="tight", pad_inches=0)
        plt.close()

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


class MaliciousPeer(Peer):
    def __init__(self, peer_id: int, balance: int, is_slow: bool, is_low_cpu: bool, interarrival_time: float, timeout: int):
        super().__init__(peer_id, balance, is_slow, is_low_cpu, interarrival_time, timeout)
        self.malicious : bool = True
        self.ringleader : bool = False
        self.ringleader_id : int = None
        self.blockchain_tree = MaliciousBlockchainTree()
        self.malicious_nodes: List[MaliciousPeer] = []
        self.malicious_neighbours: List[MaliciousPeer] = []
        self.release_counter = 0
    
    def receive_hash(self, hash: int, sender: int, timestamp: int) -> List[Event]:
        overlay = sender in [peer.peer_id for peer in self.malicious_neighbours]
        if hash not in self.hashes_seen:
            self.hashes_seen.add(hash)
            self.pending_blocks[hash].append(sender)
            ## Get Request Event to the hash sender + Timeout Event
            request = Event(timestamp, EventType.SEND_REQUEST, hash, self.peer_id, sender, overlay)
            timeout = Event(timestamp + self.timeout, EventType.TIMEOUT, hash, self.peer_id, None, overlay)
            return (request, timeout)
        else:
            if self.pending_blocks[hash]:
                ## if a timeout is already ongoing, add the hash to the queue for sending request after
                ## current timeout expires
                self.pending_blocks[hash].append(sender)
                return ()
            else:
                self.pending_blocks[hash].append(sender)
                ## Get Request Event to the hash sender + Timeout Event
                request = Event(timestamp, EventType.SEND_REQUEST, hash, self.peer_id, sender, overlay)
                timeout = Event(timestamp + self.timeout, EventType.TIMEOUT, hash, self.peer_id, None, overlay)
                return (request, timeout)

    def receive_block(self, block: Block, timestamp, sender_id: int) -> int:
        ## check if before timeout
        if not self.pending_blocks[block.hash] or self.pending_blocks[block.hash][0] != sender_id:
            return -1
        if block.hash in self.blocks_seen:
            return -1
        self.pending_blocks.pop(block.hash)
        self.blocks_seen[block.hash] = block

        ## -1: already seen block, no forwarding
        ## 1: longest chain switches terminate and re-start mining, forward block
        ## 0: forward block appropriately
        ## 2: no switching of longest chain, trigger dump of private chain

        
        release_private_chain = self.blockchain_tree.add(block, timestamp)

        if not self.ringleader: # don't start mining if not ringleader
            return 0
        
        if release_private_chain:
            return 2

        longest_chain_leaf = self.blockchain_tree.longest_chain_leaf
        if longest_chain_leaf.block.block_id == self.block_being_mined.parent_block_id:
            ## longest chain maintained
            return 0

        ## longest chain switches
        ## update self balance
        self.balance = longest_chain_leaf.balance_map[self.peer_id]
        old_parent_node = self.blockchain_tree.nodes[self.block_being_mined.parent_block_id]
        ## update mempool
        ## add the tx set in the old branch upto the LCA
        ## remove those in the new branch from LCA onwards
        old_branch_tx_set, new_branch_tx_set = self.blockchain_tree.lca_branch_txs(old_parent_node)
        mempool_set = set(self.mem_pool)
        mempool_set = mempool_set.union(old_branch_tx_set).difference(new_branch_tx_set)
        self.mem_pool = list(mempool_set)
        return 1
            
class P2PNetwork:
    def __init__(self, num_peers: int, frac_slow: bool, frac_low_cpu: bool, interarrival_time: float, I: float, frac_malicious: float, timeout: int): ## z0 is frac_slow, z1 is frac_low_cpu
        self.peers: List[Peer] = []
        self.num_peers = num_peers
        self.frac_slow = frac_slow
        self.frac_low_cpu = frac_low_cpu
        self.I = I
        self.low_hashing_power = 1 / (num_peers * (10 - 9 * frac_low_cpu))
        self.high_hashing_power = 10 / (num_peers * (10 - 9 * frac_low_cpu))
        self.interarrival_time = interarrival_time
        self.latencies = {}
        self.malicious_latencies = {}
        self.continue_simulation = True
        self.frac_malicious = frac_malicious
        self.timeout = timeout
        self.malicious_peers : Dict[int, MaliciousPeer] = {} # id -> MaliciousPeer
        self.malicious_hashing_power : float = 0
        
        self.create_peers()
        self.connect_peers()
        self.connect_malicious_peers()
        # self.print_network()
        self.initialize_latencies()
        self.initialize_malicious_latencies()
        self.initialize_event_queue()

    def print_network(self):
        ## print the network graph and malicious nodes and mark overlay and normal connections
        G = nx.Graph()

        # Add normal and malicious peers
        for peer in self.peers:
            G.add_node(peer.peer_id)
        for peer in self.malicious_peers.values():
            G.add_node(peer.peer_id)

        # Add normal edges
        normal_edges = []
        for peer in self.peers:
            for neighbour in peer.neighbours:
                normal_edges.append((peer.peer_id, neighbour.peer_id))
        
        # Add malicious edges
        malicious_edges = []
        for peer in self.malicious_peers.values():
            for neighbour in peer.malicious_neighbours:
                malicious_edges.append((peer.peer_id, neighbour.peer_id))

        pos = nx.spring_layout(G)

        # Draw nodes
        nx.draw(G, pos, with_labels=True, font_weight='bold', node_color='skyblue', node_size=500, font_size=10)
        nx.draw_networkx_nodes(G, pos, nodelist=[peer.peer_id for peer in self.malicious_peers.values()], node_color='red', node_size=500)

        # Draw normal edges in black
        nx.draw_networkx_edges(G, pos, edgelist=normal_edges, edge_color='black')

        # Draw malicious edges in red
        nx.draw_networkx_edges(G, pos, edgelist=malicious_edges, edge_color='red', style='dashed')

        plt.savefig("network.png")

    


    def initialize_event_queue(self):
        ## Start transaction generation, mining for all peers
        self.event_queue = EventQueue()
        for i in range(self.num_peers):
            sender = self.peers[i]
            gen_tx_event = sender.generate_transaction(0, self.peers)
            self.event_queue.add_event(gen_tx_event)
            ## add receive transaction events for neighbours
            if gen_tx_event.data.amount != 0:
                ## make sure transaction is not a dummy
                self.forward_packet(sender, gen_tx_event.data, gen_tx_event.timestamp, i)
            
            if not sender.malicious:
                hashing_power = self.low_hashing_power if sender.is_low_cpu else self.high_hashing_power
                end_mining_event = sender.start_mining(0, hashing_power, self.I)
            elif sender.ringleader:
                end_mining_event = sender.start_mining(0, self.malicious_hashing_power, self.I)
            else:
                continue
            self.event_queue.add_event(end_mining_event)

    def create_peers(self):
        ## choose frac_malicious * num_peers indices randomly
        malicious_indices = random.sample(range(self.num_peers), int(self.frac_malicious * self.num_peers))
        low_cpu_indices = random.sample(range(self.num_peers), int(self.frac_low_cpu * self.num_peers))
        slow_indices = random.sample(range(self.num_peers), int(self.frac_slow * self.num_peers))
        for i in range(self.num_peers):
            init_balance = 0
            is_slow = i in slow_indices
            is_low_cpu = i in low_cpu_indices
            is_malicious = i in malicious_indices
            if is_malicious:
                peer = MaliciousPeer(i, init_balance, is_slow, is_low_cpu, self.interarrival_time, self.timeout)
                self.malicious_peers[peer.peer_id] = peer
            else:
                peer = Peer(i, init_balance, is_slow, is_low_cpu, self.interarrival_time, self.timeout)
            self.peers.append(peer)
        
        ## choose random ringleader amongst malicious nodes
        ringleader = random.choice(list(self.malicious_peers.values()))
        ringleader.ringleader = True
        self.ringleader = ringleader.peer_id
        print(f"Ringleader is {ringleader.peer_id}")
        for peer in self.malicious_peers.values():
            print("Malicious peer", peer.peer_id)
            peer.malicious_nodes = list(self.malicious_peers.values())
            peer.ringleader_id = ringleader.peer_id
            self.malicious_hashing_power += self.low_hashing_power if peer.is_low_cpu else self.high_hashing_power
            peer.blockchain_tree.ringleader_id = ringleader.peer_id
    
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

            
    def connect_malicious_peers(self):
        while True:
            for peer in self.malicious_peers.values():
                peer.malicious_neighbours = []
            
            for peer in self.malicious_peers.values():
                num_connections = min(len(self.malicious_peers), random.randint(3, 6))
                while len(peer.malicious_neighbours) < num_connections:
                    neighbour = random.choice(list(self.malicious_peers.values()))
                    if neighbour != peer and neighbour not in peer.malicious_neighbours:
                        peer.malicious_neighbours.append(neighbour)
                        neighbour.malicious_neighbours.append(peer)
            
            if self.check_malicious_connectivity():     
                break

    def check_connectivity(self):
        ## does BFS to check graph connectedness
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
    
    def check_malicious_connectivity(self):
        ## does BFS to check graph connectedness
        visited = {peer.peer_id: False for peer in self.malicious_peers.values()}
        stack = [list(self.malicious_peers.keys())[0]]
        visited[stack[0]] = True
        count = 1
        while stack:
            peer_id = stack.pop()
            for neighbour in self.malicious_peers[peer_id].malicious_neighbours:
                if not visited[neighbour.peer_id]:
                    visited[neighbour.peer_id] = True
                    stack.append(neighbour.peer_id)
                    count += 1
        return count == len(self.malicious_peers)
    
    def initialize_latencies(self):
        for peer in self.peers:
            for neighbor in peer.neighbours:
                if (peer.peer_id, neighbor.peer_id) not in self.latencies:
                    rho = random.uniform(10, 500)  # Propagation delay in ms
                    link_speed = 100 if not (peer.is_slow or neighbor.is_slow) else 5  # 100 Mbps or 5 Mbps
                    self.latencies[(peer.peer_id, neighbor.peer_id)] = {
                        "rho": rho / 1000,  # Convert to seconds
                        "c": link_speed/8,
                    }
                    self.latencies[(neighbor.peer_id, peer.peer_id)] = self.latencies[(peer.peer_id, neighbor.peer_id)]
    
    def initialize_malicious_latencies(self):
        for peer in self.malicious_peers.values():
            for neighbour in peer.malicious_neighbours:
                if (peer.peer_id, neighbour.peer_id) not in self.malicious_latencies:
                    rho = random.uniform(1, 10)
                    link_speed = 100 if not (peer.is_slow or neighbour.is_slow) else 5
                    self.malicious_latencies[(peer.peer_id, neighbour.peer_id)] = {
                        "rho": rho / 1000,
                        "c": link_speed/8,
                    }
                    self.malicious_latencies[(neighbour.peer_id, peer.peer_id)] = self.malicious_latencies[(peer.peer_id, neighbour.peer_id)]        
        
    def forward_packet(self, peer: Peer, data: Union[Transaction, Block, int, str], timestamp, sender_id: int, overlay: bool = False):
        ## loopless forwarding with random latency for packets (transactions, blocks)
        if overlay: # choose network to forward packet
            latencies_map = self.malicious_latencies
            neighbours = peer.malicious_neighbours
        else:
            latencies_map = self.latencies
            neighbours = peer.neighbours
        for neighbour in neighbours:
            if neighbour.peer_id == sender_id:
                continue
            ## ensuring that only overlay communication between connected malicious nodes
            if not overlay and peer.malicious and neighbour.malicious:
                continue
            latencies = latencies_map[(peer.peer_id, neighbour.peer_id)]
            rho = latencies["rho"]
            c = latencies["c"]
            d = random.expovariate((12/1024) / c)
            if isinstance(data, Transaction):
                message_size = 1
                event_type = EventType.RECEIVE_TRANSACTION
            elif isinstance(data, int):
                message_size = 1/16 ## Hash size is 64B = 64/1024 KB
                event_type = EventType.RECEIVE_HASH
            elif isinstance(data, str): ## Pvt Chain release msg 64B
                message_size = 1/16 
                event_type = EventType.RELEASE_PRIVATE_CHAIN
            else:
                message_size = len(data.transactions)
                event_type = EventType.RECEIVE_BLOCK
            ## number of KBs
            receiver_delay = rho + message_size * TX_SIZE / c + d
            ## make the neighbour receive the packet with random delay
            event = Event(timestamp + receiver_delay, event_type, data, peer.peer_id, neighbour.peer_id, overlay)
            self.event_queue.add_event(event)

    def forward_packet_single(self, sender: Peer, data: Union[Transaction, Block, int], timestamp, receiver_id: int, event_type, overlay: bool = False):
        # assert self.peers[receiver_id] in sender.neighbours and receiver_id != sender.peer_id

        if not overlay and sender.malicious and self.peers[receiver_id].malicious:
            return
        
        if overlay:
            latencies = self.malicious_latencies[(sender.peer_id, receiver_id)]
        else:
            latencies = self.latencies[(sender.peer_id, receiver_id)]
        rho = latencies["rho"]
        c = latencies["c"]
        d = random.expovariate((12/1024) / c)
        if isinstance(data, Transaction):
            message_size = 1
        elif isinstance(data, int):
            message_size = 1/16 ## Hash size is 64B = 64/1024 KB
        else:
            message_size = len(data.transactions)
        ## number of KBs
        receiver_delay = rho + message_size * TX_SIZE / c + d
        ## make the neighbour receive the packet with random delay
        event = Event(timestamp + receiver_delay, event_type, data, sender.peer_id, receiver_id, overlay)
        self.event_queue.add_event(event)


    def process_events(self, output_dir: str, stopping_height: int, suppress_output: bool, stopping_time: int):
        ## process the queue
        while self.continue_simulation:
            while self.event_queue.queue:
                event = self.event_queue.pop_event()
                if event.timestamp >= stopping_time:
                    self.write_balances(f"{output_dir}/balances.txt")
                    self.write_ratios(f"{output_dir}/ratios.txt")
                    for peer in self.peers:
                        if not suppress_output:
                            peer.write_to_file(f"{output_dir}/tree_peer_{peer.peer_id}.txt")
                            peer.visualize_tree(output_dir = output_dir, ringleader_id = self.ringleader)
                    print(f"Simulation time {event.timestamp} exceeded stopping time {stopping_time}. Exiting simulation...")
                    self.continue_simulation = False
                    break
                elif event.event_type == EventType.GENERATE_TRANSACTION:
                    self.process_generate_transaction(event)
                elif event.event_type == EventType.RECEIVE_TRANSACTION:
                    self.process_receive_transaction(event)
                elif event.event_type == EventType.END_MINING:
                    self.process_end_mining(event)
                elif event.event_type == EventType.RECEIVE_HASH:
                    self.process_receive_hash(event)
                elif event.event_type == EventType.SEND_REQUEST:
                    self.process_send_request(event)
                elif event.event_type == EventType.RECEIVE_REQUEST:
                    self.process_receive_request(event)
                elif event.event_type == EventType.RECEIVE_BLOCK:
                    self.process_receive_block(event)
                elif event.event_type == EventType.TIMEOUT:
                    self.process_timeout(event)
                elif event.event_type == EventType.RELEASE_PRIVATE_CHAIN:
                    if DEBUG:
                        print("releasing chain")
                    self.process_release(event)
                continue
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
        self.event_queue.add_event(gen_tx_event) ## add to event queue to recursively generate next transaction
        if gen_tx_event.data.amount != 0:
            ## make sure transaction is not a dummy
            self.forward_packet(peer, gen_tx_event.data, gen_tx_event.timestamp, id)
            if peer.malicious:
                self.forward_packet(peer, gen_tx_event.data, gen_tx_event.timestamp, id, True)

    def process_receive_transaction(self, event: Event):
        receiver = self.peers[event.receiver]
        transaction = event.data
        unseen_tx = receiver.receive_transaction(transaction)
        if unseen_tx:
            ## forward only if new tx
            self.forward_packet(receiver, transaction, event.timestamp, event.sender)
            if receiver.malicious:
                self.forward_packet(receiver, transaction, event.timestamp, event.sender, True)

    def process_end_mining(self, event: Event):
        id = event.sender
        peer = self.peers[id]
        block = event.data
        if peer.malicious:
            assert peer.ringleader
            hashing_power = self.malicious_hashing_power
            if DEBUG:
                print(f"Ringleader mined block {block.block_id}")
                print(f"Block is mined by {block.get_miner_id()}")
        else:
            hashing_power = self.low_hashing_power if peer.is_low_cpu else self.high_hashing_power
        if block.block_id != peer.block_being_mined.block_id:
            ## do not mine the block if the chain has switched in the mean time
            if DEBUG:
                print(f"Block being mined {peer.block_being_mined.block_id}")
            return 
        ## mine the block, and forward to neighbours
        peer.mine(block, event.timestamp) 
        self.forward_packet(peer, block.hash, event.timestamp, id, peer.malicious)
        ## start mining all over again
        end_mining_event = peer.start_mining(event.timestamp, hashing_power, self.I)
        self.event_queue.add_event(end_mining_event)

    def process_receive_hash(self, event: Event):
        receiver = self.peers[event.receiver]
        hash = event.data
        events = receiver.receive_hash(hash, event.sender, event.timestamp)
        for event_ in events:
            self.event_queue.add_event(event_)

    def process_receive_block(self, event: Event):
        receiver = self.peers[event.receiver]
        sender_id = event.sender
        block = event.data
        if receiver.malicious: # decide hashing power depending on type of node
            hashing_power = self.malicious_hashing_power
        else:
            hashing_power = self.low_hashing_power if receiver.is_low_cpu else self.high_hashing_power

        return_code = receiver.receive_block(block, event.timestamp, sender_id)

        if return_code != -1: ## unseen block
            self.forward_packet(receiver, block.hash, event.timestamp, event.sender, receiver.malicious)
            if not (event.overlay and block.get_miner_id() == self.ringleader):
                self.forward_packet(receiver, block.hash, event.timestamp, event.sender, False)

        if return_code == 1:
            ## new longest chain formed, re-start mining on the same
            event = receiver.start_mining(event.timestamp, hashing_power, self.I)
            self.event_queue.add_event(event)

        if return_code == 2:
            assert receiver.ringleader
            ## release private chain, do not re-start mining as already mining
            ## set release msg counter, add release event for ringleader
            self.forward_packet(receiver, f"{receiver.release_counter + 1}", event.timestamp, self.ringleader, True)
            event = Event(event.timestamp, EventType.RELEASE_PRIVATE_CHAIN, f"{receiver.release_counter + 1}", self.ringleader, self.ringleader, True)
            self.event_queue.add_event(event)
            

    ## honest peer - sends out hash to everyone else on normal network
    ## malicious peer - 
    ##                - receives on normal network - forward hash to malicious on overlay send hash to honest on normal
    ##                - receives on overlay - if its a malicious block, forward hash on overlay
    ##                                      - if its a honest block, forward hash to malicious on overlay, send hash to honest on normal
    

    def process_send_request(self, event: Event):
        requester = self.peers[event.sender]
        provider = event.receiver
        ## event.overlay == requester and provider in self.malicious_neighbours
        if DEBUG:
            print(f"Sender {event.sender} sending request to provider {event.receiver}, overlay {event.overlay}")
        self.forward_packet_single(requester, event.data, event.timestamp, provider, EventType.RECEIVE_REQUEST, event.overlay)
        

    def process_receive_request(self, event: Event):
        ## The event's receiver is going to process the request
        provider = self.peers[event.receiver]
        hash = event.data
        if provider.malicious and not event.overlay:
            if provider.blocks_seen[hash].get_miner_id() == self.ringleader: ## honest neighbour asking for malicious block
                self.forward_packet_single(provider, provider.blocks_seen[hash], event.timestamp, event.sender, EventType.RECEIVE_BLOCK, False)
            else: ## honest neighbour asking for honest block
                # return
                ## to remove eclipse attack
                self.forward_packet_single(provider, provider.blocks_seen[hash], event.timestamp, event.sender, EventType.RECEIVE_BLOCK, False)
                return
        block = provider.receive_request(hash)
        if DEBUG:
            print(f"Sender {event.sender} received block from provider {event.receiver}")
        if block:
            self.forward_packet_single(provider, block, event.timestamp, event.sender, EventType.RECEIVE_BLOCK, event.overlay)
    
    def process_timeout(self, event: Event):
        requester = self.peers[event.sender]
        hash = event.data
        events = requester.process_timeout(hash, event.timestamp)
        for event_ in events:
            self.event_queue.add_event(event_)

    def process_release(self, event: Event):
        receiver = self.peers[event.receiver]
        assert receiver.malicious
        if int(event.data) <= receiver.release_counter:
            # loopless forwarding of release broadcast msg by release msg counter
            return
        receiver.release_counter = int(event.data) 
        # update counter to latest pvt chain release msg

        ## RELEASE THE CHAIN - FORWARD HASHES IN HONEST NETWORK
        private_chain : List[Block] = receiver.blockchain_tree.get_private_chain()
        if DEBUG:
            print(f"Length of pvt chain {len(private_chain)}")
            # if len(private_chain) == 0:
            # check that the miner_id of all blocks in the private chain is the ringleader
            if len(private_chain) > 0 and not all(block.get_miner_id() == receiver.ringleader_id for block in private_chain):
                print("Not all malicious in private chain")
                print([block.get_miner_id() for block in private_chain])
                for dict_ in receiver.pending_blocks.values():
                    print(dict_)
                receiver.visualize_tree(output_dir = "results", ringleader_id = self.ringleader)
                exit()
        
        for block in private_chain: # release all block hash from pvt chain
            self.forward_packet(receiver, block.hash, event.timestamp, receiver.peer_id, False)

        # forward the release broadcast msg to malicious nodes on the overlay network
        self.forward_packet(receiver, event.data, event.timestamp, event.sender, True)

    def write_balances(self, file_name: str):
        for peer in self.peers:
            balance_map = peer.blockchain_tree.longest_chain_leaf.balance_map
            # sort balance map by keys
            sorted_balance_map = dict(sorted(balance_map.items()))
            with open(file_name, 'a') as file:
                file.write(f"Peer {peer.peer_id} balance map: {sorted_balance_map}\n")
                file.write(f"Sum balance map {sum(sorted_balance_map.values())}\n")
    
    def print_blockchain_tree_height(self):
        for peer in self.peers:
            print(f"Peer {peer.peer_id} has blockchain height {peer.blockchain_tree.longest_chain_leaf.height}")
        
    def write_ratios(self, file_name: str):
        ratio_map = {'slow_low': [], 'slow_high': [], 'fast_low': [], 'fast_high': []}
        peer_ratios = []
        for peer in self.peers:
            peer_ratio = peer.get_ratio()
            if peer.is_slow and peer.is_low_cpu:
                ratio_map['slow_low'].append(peer_ratio)
            elif peer.is_slow and not peer.is_low_cpu:
                ratio_map['slow_high'].append(peer_ratio)
            elif not peer.is_slow and peer.is_low_cpu:
                ratio_map['fast_low'].append(peer_ratio)
            else:
                ratio_map['fast_high'].append(peer_ratio)
            network_metadata = "slow" if peer.is_slow else "fast"
            cpu_metadata = "low" if peer.is_low_cpu else "high"
            ratio_string = f"Peer {peer.peer_id} {network_metadata} {cpu_metadata} ratio: {peer_ratio[0]}, blocks in longest chain: {peer_ratio[1]}, blocks mined: {peer_ratio[2]}"
            peer_ratios.append(ratio_string)
        
        for key in ratio_map:
            ratios = [row[0] for row in ratio_map[key]]
            my_blocks_in_longest_chain = [row[1] for row in ratio_map[key]]
            blocks_mined = [row[2] for row in ratio_map[key]]
            # write to file_name
            with open(file_name, 'a') as file:
                avg_ratio = sum(ratios) / len(ratios) if len(ratios) != 0 else 0
                avg_blocks_longest_chain = sum(my_blocks_in_longest_chain) / len(my_blocks_in_longest_chain) if len(my_blocks_in_longest_chain) != 0 else 0
                avg_blocks_mined = sum(blocks_mined) / len(blocks_mined) if len(blocks_mined) != 0 else 0
                
                file.write(f"Average ratio for {key}: {avg_ratio}\n")
                file.write(f"Average blocks in longest chain for {key}: {avg_blocks_longest_chain}\n")
                file.write(f"Average blocks mined for {key}: {avg_blocks_mined}\n")
        
        for peer_data in peer_ratios:
            with open(file_name, 'a') as file:
                file.write(peer_data + "\n")

    def write_metrics(self, output_dir: str):
        ringleader = self.peers[self.ringleader]
        red_total_blocks = ringleader.blocks_mined
        longest_chain = ringleader.blockchain_tree.longest_chain_leaf

        ## traverse the longest chain till genesis block
        chain = []
        curr_node = longest_chain
        while curr_node:
            chain.append(curr_node)
            curr_node = curr_node.parent
        
        len_longest_chain = len(chain)
        red_blocks_in_longest_chain = sum([1 for node in chain if node.miner_id == self.ringleader])

        with open(f"{output_dir}/metrics.txt", 'w') as file:
            file.write(f"Total blocks mined by ringleader: {red_total_blocks}\n")
            file.write(f"Total blocks in longest chain: {len_longest_chain}\n")
            file.write(f"Total red blocks in longest chain: {red_blocks_in_longest_chain}\n")
            file.write(f"Red blocks in longest chain by longest chain length: {red_blocks_in_longest_chain / len_longest_chain}\n")
            file.write(f"Red blocks in longest chain by total red blocks: {red_blocks_in_longest_chain / red_total_blocks}\n")

                
    
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='P2P Network Simulator')
    parser.add_argument('-num_peers', type=int, help='Number of peers in the network', required=True)
    parser.add_argument('-frac_slow', type=float, help='Fraction of slow peers', required=True)
    parser.add_argument('-frac_low_cpu', type=float, help='Fraction of low CPU peers', required=True)
    parser.add_argument('-interarrival_time', type=float, help='Mean interarrival time of transactions', required=True)
    parser.add_argument('-I', type=float, help='Block mining time', required=True)
    parser.add_argument('-stopping_height', type=int, help='Simulation stopping criterion', default=10)
    parser.add_argument('-suppress_output', dest='suppress_output', help='Verbose, logs, plots', action='store_true')
    parser.add_argument('-stopping_time', type=int, help='Simulation stopping criterion', default=10000)
    parser.add_argument('-frac_malicious', type=float, help='Fraction of Malicious nodes', default=0)
    parser.add_argument('-timeout', type=int, help='Timeout for Get Requests for blocks', default=500)
    parser.set_defaults(suppress_output=False)
    

    args = parser.parse_args()
    num_peers = args.num_peers
    frac_slow = args.frac_slow
    frac_low_cpu = args.frac_low_cpu
    interarrival_time = args.interarrival_time
    stopping_height = args.stopping_height
    suppress_output = args.suppress_output
    stopping_time = args.stopping_time
    frac_malicious = args.frac_malicious
    timeout = args.timeout
    I = args.I
    output_dir = f"non_eclipse/results_{frac_malicious}_{timeout}"
    os.makedirs(output_dir, exist_ok = True)
    network = P2PNetwork(num_peers, frac_slow, frac_low_cpu, interarrival_time, I, frac_malicious, timeout)

    network.process_events(output_dir = output_dir, stopping_height = stopping_height, suppress_output = suppress_output, stopping_time = stopping_time)

    network.write_metrics(output_dir)

