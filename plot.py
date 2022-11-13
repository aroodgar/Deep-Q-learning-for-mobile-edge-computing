import matplotlib.pyplot as plt
from random import randint
import numpy as np
import math

class Plot:
  def __init__(self, plots, names): # plots is a list of dicts
    self.plots = plots
    self.names = names
  
  def draw_plot(self):
    n = int(len(self.names) / 2)
    r = len(self.names) % 2
    if r > 0:
      n += 1

    index = 0
    for plot in self.plots:
      plt.clf()
      plt.plot(plot['x'], plot['y'], label=self.names[index])
      plt.savefig(f'{self.names[index]}.png')
      
      index += 1

def test():    
  X = np.arange(0, math.pi*2, 0.05)
  Y1 = np.sin(X)
  Y2 = np.cos(X)
  Y3 = np.tan(X)
  Y4 = np.tanh(X)
  plots = [{'x': X, 'y': Y1},
  {'x': X, 'y': Y2},
  {'x': X, 'y': Y3},
  {'x': X, 'y': Y4}]
  names = [
    'Sine Function',
    'Cosine Function',
    'Tangent Function',
    'Tanh Function'
  ]

  p = Plot(plots, names)
  p.draw_plot()

if __name__ == '__main__':
  test()
