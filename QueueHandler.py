from enum import Enum
from GeneralQueue import FIFO, QueueType1, QueueType2

class QueueHandler:
  def __init__(self, queue_name):
    self.queue = class_dict[queue_name]()
    print(f"Queue Type = {type(self.queue)}")
  
  def isEmpty(self):
    return self.queue.isEmpty()

  def put(self, item):
    self.queue.put(item)

  def get(self):
    return self.queue.get()

class Queues(Enum):
  FIFO = 1
  TYPE1 = 2
  TYPE2 = 3

class_dict = {
  '{}'.format(Queues.FIFO.name): FIFO,
  '{}'.format(Queues.TYPE1.name): QueueType1,
  '{}'.format(Queues.TYPE2.name): QueueType2
}