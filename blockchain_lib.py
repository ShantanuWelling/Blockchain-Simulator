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
        balance_map = balances.copy()
        for tx in sorted(transactions):
            if tx.sender == None:
                if tx.amount != 50:
                    return False
            else:        
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
    def __init__(self, block: Block, parent: "BlockChainNode"):
        self.block = block
        self.children: list[BlockChainNode] = []
        self.parent = parent
        self.height = self.parent.height + 1 if self.parent else 0
        self.balance_map: Dict[int, int] = {}
        ## update balance map
        if self.parent:
            self.balance_map = self.parent.balance_map.copy()
            for tx in block.transactions:
                if tx.sender is not None:
                    self.balance_map[tx.sender] -= tx.amount
                self.balance_map[tx.receiver] += tx.amount
        else:
            self.balance_map = defaultdict(int) ## SW

        def __eq__(node1: "BlockChainNode", node2: "BlockChainNode") -> bool:
            return node1.block.block_id == node2.block.block_id

class BlockchainTree:
    def __init__(self):
        self.genesis_block = Block(0, [], -1, 0)
        self.root = BlockChainNode(self.genesis_block, None)
        self.buffer: list[Block] = [] # Buffer for blocks that are not yet part of the blockchain
        self.nodes: Dict[int, BlockChainNode] = {0: self.root}
        self.longest_chain_leaf = self.root
        self.longest_chain_txs: Set["Transaction"] = set()
        self.height = 0

    def lca_branch_txs(self, node: BlockChainNode):
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
        block_branch_txs = set(sum([node.block.transactions for node in block_chain], []))
        longest_branch_txs = set(sum([node.block.transactions for node in longest_chain], []))
        return (block_branch_txs, longest_branch_txs)

    def chain_transactions(self, node: BlockChainNode) -> Set[Transaction]:
        if node.parent == self.longest_chain_leaf:
            return self.longest_chain_txs.union(set(node.block.transactions))
        txs_to_add, txs_to_remove = self.lca_branch_txs(node)
        return self.longest_chain_txs.union(txs_to_add).difference(txs_to_remove)
    
    def add(self, block: Block) -> Block:
        self.buffer.append(block)
        prev_buffer_len = -1

        while len(self.buffer) != prev_buffer_len:
            prev_buffer_len = len(self.buffer)
            # print("while", len(self.buffer))
            for block in self.buffer:
                if block.parent_block_id not in self.nodes:
                    # print("b1")
                    continue
                # print("here1")
                parent_node = self.nodes[block.parent_block_id]

                ## TODO: need to remove blocks from the buffer that are children of an invalid block

                ## Time check
                if block.create_timestamp < parent_node.block.create_timestamp:
                    self.buffer.remove(block)
                    # print("b2")
                    continue
                ## Balance check
                valid_block = validate(block.transactions, parent_node.balance_map)
                if not valid_block:
                    self.buffer.remove(block)
                    # print("b3")
                    continue
                ## New transactions check
                chain_transactions = self.chain_transactions(parent_node)
                if chain_transactions.intersection(set(block.transactions)):
                    self.buffer.remove(block)
                    # print("b4")
                    continue

                ## Add tx_ids of block in own seen_txs if added on longest chain?
                ## update of Peer's mem_pool done in mine, and in receive block
                # print('update')
                parent_node = self.nodes[block.parent_block_id]
                new_node = BlockChainNode(block, parent_node)
                self.nodes[block.block_id] = new_node
                parent_node.children.append(new_node)
                self.buffer.remove(block)
                longest_chain_timestamp = self.longest_chain_leaf.block.create_timestamp
                if new_node.height > self.height or new_node.height == self.height and longest_chain_timestamp > block.create_timestamp:
                    ## Switch chain if same length but the new block is chronologically earlier than old?
                    self.longest_chain_txs = self.chain_transactions(new_node)
                    self.height = new_node.height
                    self.longest_chain_leaf = new_node

        
            
    

