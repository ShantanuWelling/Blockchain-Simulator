import heapq
from enum import Enum, auto
from typing import Union, Dict

class EventType(Enum):
    GENERATE_TRANSACTION = auto()
    RECEIVE_TRANSACTION = auto()
    END_MINING = auto()
    RECEIVE_BLOCK = auto()
    
COINBASE_REWARD = 50  # 50 coins

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
    def __init__(self, block_id, transactions: list[Transaction], parent_block_id, timestamp):
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
            self.balance_map = {tx.sender: 0 for tx in block.transactions} ## SW

class BlockchainTree:
    def __init__(self):
        self.genesis_block = Block(0, [], -1)
        self.root = BlockChainNode(self.genesis_block, None)
        self.buffer: list[Block] = [] # Buffer for blocks that are not yet part of the blockchain
        self.nodes: Dict[int, BlockChainNode] = {0: self.root}
        self.longest_chain_leaf = self.root
        self.height = 0

    def add(self, block: Block) -> Block:
        self.buffer.append(block)
        prev_buffer_len = -1
        while len(self.buffer) != prev_buffer_len:
            prev_buffer_len = len(self.buffer)
            for block in self.buffer:
                if block.parent_block_id not in self.nodes:
                    continue
                ## validate block, if failed, remove from buffer, continue, else add to blockchain
                ## check if coinbase transaction is valid
                if block.transactions[0].amount != COINBASE_REWARD  # 50 coins
                    remove_block_and_its_children(block)
                    continue
                ## check if tranactions in this block are not there in the chain in which it is being added
                for tx in block.transactions:

                    ## check if final balances dont go negative for any peer
                    ## check if block timestamp is greater than parent block timestamp
                

                ## add tx_ids of block in own seen_txs if added on longest chain and remove those from our mempool


                parent_node = self.nodes[block.parent_block_id]
                new_node = BlockChainNode(block, parent_node)
                self.nodes[block.block_id] = new_node
                parent_node.children.append(new_node)
                if new_node.height > self.height:
                    self.height = new_node.height
                    self.longest_chain_leaf = new_node
            self.buffer = [block for block in self.buffer if block.block_id not in self.nodes]

        
            
    

