class Config:

  def __init__(self,
    NUM_IOT=50, NUM_FOG=5,
    NUM_EPISODE=1000, NUM_TIME_BASE=100,
    MAX_DELAY=10, PARAM_UPDATE_FREQ=200,
    LR=0.001, BATCH_SIZE=16,
    TASK_ARRIVE_PROB=0.3, QUEUE_TYPE=None):

    self.num_iot = NUM_IOT
    self.num_fog = NUM_FOG
    self.num_episode = NUM_EPISODE
    self.num_time_base = NUM_TIME_BASE
    self.max_delay = MAX_DELAY
    self.param_update_freq = PARAM_UPDATE_FREQ
    self.lr = LR
    self.batch_size = BATCH_SIZE
    self.task_arrive_prob = TASK_ARRIVE_PROB
    self.queue_type = QUEUE_TYPE


  

