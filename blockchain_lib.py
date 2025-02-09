import heapq
from enum import Enum, auto
from typing import Union, Dict, List, Set
from collections import defaultdict

class EventType(Enum):
    GENERATE_TRANSACTION = auto()
    RECEIVE_TRANSACTION = auto()
    END_MINING = auto()
    RECEIVE_BLOCK = auto()
    
COINBASE_REWARD = 50  # 50 coins

def validate(transactions: List["Transaction"], balances: Dict[int, int]) -> bool:
    ## Validates a list of transaction given a balance map of the peers
        balance_map = balances.copy()
        for tx in sorted(transactions):
            ## sorting done according to the transaction timestamps
            if tx.sender == None:
                ## check coinbase transaction amount
                if tx.amount != 50:
                    return False
            else:
                ## insufficient sender balance for the transaction       
                if balance_map[tx.sender] < tx.amount:
                    return False
                balance_map[tx.sender] -= tx.amount
            balance_map[tx.receiver] += tx.amount
        return True

class Transaction:
    def __init__(self, tx_id, sender: int, receiver: int, amount: int, timestamp):
        self.tx_id = tx_id
        self.sender = sender
        self.receiver = receiver
        self.amount = amount
        self.timestamp = timestamp
    
    def __str__(self):
        if self.sender is None:
            return f"{self.tx_id}: {self.receiver} mines {self.amount} coins"
        return f"{self.tx_id}: {self.sender} pays {self.receiver} {self.amount} coins"
    
    def __eq__(self, other: "Transaction"):
        if not isinstance(other, Transaction):
            return False
        return self.tx_id == other.tx_id
    
    def __hash__(self):
        return hash(self.tx_id)
    
    def __lt__(self, other: "Transaction"):
        return self.timestamp < other.timestamp
    

class Event:
    def __init__(self, timestamp, event_type: EventType, data: Union[Transaction, 'Block'], sender: int, receiver: int):
        self.timestamp = timestamp
        self.event_type = event_type
        self.data = data
        self.sender = sender
        self.receiver = receiver
    
    def __lt__(self, other: "Event"):
        return self.timestamp < other.timestamp


class EventQueue:
    ## Queue of events sorted according to the timestamps
    def __init__(self):
        self.queue: list[Event] = []
    
    def add_event(self, event: Event):
        heapq.heappush(self.queue, event)
    
    def pop_event(self) -> Event:
        return heapq.heappop(self.queue)
    

class Block:
    def __init__(self, block_id, transactions: List[Transaction], parent_block_id, timestamp):
        self.block_id = block_id
        self.transactions = transactions
        self.create_timestamp = timestamp
        self.parent_block_id = parent_block_id

class BlockChainNode:
    ## Wrapper class around a block with information required to maintain the tree for each peer
    def __init__(self, block: Block, parent: "BlockChainNode", timestamp):
        self.block = block
        self.children: list[BlockChainNode] = []
        self.parent = parent
        self.height = self.parent.height + 1 if self.parent else 0
        ## cumulative balance of each peer according to the blockchain upto this node
        self.balance_map: Dict[int, int] = {}
        self.miner_id: int = -1
        self.receive_timestamp = timestamp
        if block.transactions:
            ## ID of the peer in the coinbase transaction
            self.miner_id = block.transactions[0].receiver
        ## update balance map using parent's map
        if self.parent:
            self.balance_map = self.parent.balance_map.copy()
            for tx in block.transactions:
                if tx.sender is not None:
                    self.balance_map[tx.sender] -= tx.amount
                self.balance_map[tx.receiver] += tx.amount
        else:
            self.balance_map = defaultdict(int)

        def __eq__(node1: "BlockChainNode", node2: "BlockChainNode") -> bool:
            return node1.block.block_id == node2.block.block_id

class BlockchainTree:
    def __init__(self):
        self.genesis_block = Block(0, [], -1, 0)
        self.root = BlockChainNode(self.genesis_block, None, None)
        self.buffer: list[Block] = [] # buffer for blocks that are not yet part of the blockchain
        self.nodes: Dict[int, BlockChainNode] = {0: self.root}
        self.longest_chain_leaf = self.root # leaf at the end of the longest chain
        self.longest_chain_txs: Set["Transaction"] = set() # cumulative set of transactions in the longest chain 
        self.height = 0

    def lca_branch_txs(self, node: BlockChainNode):
        ## Finds the least common ancestor of given nodes in the longest chain
        ## Returns the transaction sets of the 2 branches from this LCA onwards 
        longest_chain = []
        block_chain = []
        curr_node = self.longest_chain_leaf
        while curr_node:
            longest_chain.append(curr_node)
            curr_node = curr_node.parent
        longest_chain.reverse()
        curr_node = node
        while curr_node:
            block_chain.append(curr_node)
            curr_node = curr_node.parent
        block_chain.reverse()

        while len(block_chain) > 0 and block_chain[0] == longest_chain[0]:
            longest_chain.pop(0)
            block_chain.pop(0)
        ## collect the transactions in the 2 branches
        block_branch_txs = set(sum([node.block.transactions for node in block_chain], []))
        longest_branch_txs = set(sum([node.block.transactions for node in longest_chain], []))
        return (block_branch_txs, longest_branch_txs)

    def chain_transactions(self, node: BlockChainNode) -> Set[Transaction]:
        ## Returns the set of transactions in the chain ending at the given node
        ## Used in validating a new node added at any point in the tree
        if node.parent == self.longest_chain_leaf:
            return self.longest_chain_txs.union(set(node.block.transactions))
        txs_to_add, txs_to_remove = self.lca_branch_txs(node)
        ## the tx set is built efficiently by avoiding the chain up to the LCA in the longest chain
        return self.longest_chain_txs.union(txs_to_add).difference(txs_to_remove)
    
    def add(self, block: Block, timestamp) -> Block:
        ## Adds a new block to the tree, or to the buffer if the parent is not a part of it yet
        self.buffer.append(block)
        prev_buffer_len = -1

        while len(self.buffer) != prev_buffer_len:
            prev_buffer_len = len(self.buffer)
            for block in self.buffer:
                if block.parent_block_id not in self.nodes:
                    continue

                parent_node = self.nodes[block.parent_block_id]
                ## TODO: need to remove blocks from the buffer that are children of an invalid block
                ## check that the block was created after the parent block
                if block.create_timestamp < parent_node.block.create_timestamp:
                    self.buffer.remove(block)
                    continue
                ## check that the transactions in the block are consistent with the chain it is being added to
                valid_block = validate(block.transactions, parent_node.balance_map)
                if not valid_block:
                    self.buffer.remove(block)
                    continue
                ## check that the block contains no transaction already a part of the chain
                chain_transactions = self.chain_transactions(parent_node)
                if chain_transactions.intersection(set(block.transactions)):
                    self.buffer.remove(block)
                    continue

                new_node = BlockChainNode(block, parent_node, timestamp)
                self.nodes[block.block_id] = new_node
                parent_node.children.append(new_node)
                self.buffer.remove(block)

                longest_chain_timestamp = self.longest_chain_leaf.block.create_timestamp
                ## switch the longest chain leaf to the new_node 
                if new_node.height > self.height or new_node.height == self.height and longest_chain_timestamp > block.create_timestamp:
                    self.longest_chain_txs = self.chain_transactions(new_node)
                    self.height = new_node.height
                    self.longest_chain_leaf = new_node

    def number_of_peer_blocks(self, peer_id: int) -> int:
        ## Traverses the longest chain and returns the number of blocks in it mined by the given peer
        curr_node = self.longest_chain_leaf
        number_of_blocks: int = 0
        while curr_node:
            if curr_node.miner_id == peer_id:
                number_of_blocks += 1
            curr_node = curr_node.parent
        return number_of_blocks
             
            
    

