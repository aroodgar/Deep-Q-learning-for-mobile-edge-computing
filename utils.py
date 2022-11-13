class Utils:
  def __init__(self):
    self.reset()

  def reset(self):
    self.dropped_tasks = 0
    self.total_tasks = 0
    self.total_trans_tasks_p = [0] * 3
    self.total_comp_tasks = 0
    self.total_fog_tasks = 0
    self.dropped_trans_tasks = 0
    self.dropped_trans_tasks_p = [0] * 3
    self.dropped_comp_tasks = 0
    self.dropped_fog_tasks = 0

    self.done_comp_tasks_count = 0
    self.done_fog_tasks_count = 0
    self.done_tasks = 0

    # in time slot units
    self.sum_comp_proc_delay = 0
    self.sum_fog_proc_delay = 0

  def add_task(self):
    self.total_tasks += 1

  def add_trans_task(self, p):
    self.total_trans_tasks_p[p] += 1

  def add_comp_task(self):
    self.total_comp_tasks += 1

  def add_fog_task(self):
    self.total_fog_tasks += 1

  def drop_task(self):
    self.dropped_tasks += 1

  def drop_trans_task(self, p):
    self.dropped_trans_tasks += 1
    self.dropped_trans_tasks_p[p] += 1

  def drop_comp_task(self):
    self.dropped_comp_tasks += 1

  def drop_fog_task(self):
    self.dropped_fog_tasks += 1

  def done_comp_task(self, proc_delay):
    self.done_comp_tasks_count += 1
    self.sum_comp_proc_delay += proc_delay
    self.done_task()

  def done_fog_task(self, proc_delay):
    self.done_fog_tasks_count += 1
    self.sum_fog_proc_delay += proc_delay
    self.done_task()

  def done_task(self):
    self.done_tasks += 1

  def get_all_proc_delays_sum(self):
    return self.sum_fog_proc_delay + self.sum_comp_proc_delay  

  def get_dropped_task_ratio(self):
    return self.dropped_tasks / self.total_tasks
    