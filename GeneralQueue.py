import queue

class GeneralQueue:
  def __init__(self):
    pass

  def put(self, item):
    pass

  def get(self):
    pass

  def isEmpty(self):
    pass

class FIFO(GeneralQueue):
  def __init__(self):
    self.queue = queue.Queue()

  def put(self, item):
    self.queue.put(item)

  def get(self):
    return self.queue.get()

  def isEmpty(self):
    return self.queue.empty()

class QueueType1(GeneralQueue):
  def __init__(self):
    pass

  def put(self, item):
    pass
  
  def get(self):
    pass

  def isEmpty(self):
    pass

class QueueType2(GeneralQueue):
    def __init__(self):
      pass
 
    def isEmpty(self):
      pass
 
    def put(self, p_item):
      pass
 
    def get(self):
      pass
