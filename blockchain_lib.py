import heapq
from enum import Enum, auto

class EventType(Enum):
    GENERATE_TRANSACTION = auto()
    RECEIVE_TRANSACTION = auto()
    
    
class Transaction:
    def __init__(self, tx_id, sender: int, receiver: int, amount: int):
        self.tx_id = tx_id
        self.sender = sender
        self.receiver = receiver
        self.amount = amount
    
    def __str__(self):
        return f"{self.tx_id}: {self.sender} pays {self.receiver} {self.amount} coins"
    
    def __eq__(self, other: "Transaction"):
        if not isinstance(other, Transaction):
            return False
        return self.tx_id == other.tx_id
    
    def __hash__(self):
        return hash(self.tx_id)
    

class Event:
    def __init__(self, timestamp, event_type: EventType, data: Transaction, sender: int, receiver: int):
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
    

    