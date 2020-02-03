
#!/usr/bin/env python
#--------------------
""" Math methods for numpy arrays.
""" 

import numpy as np
from math import sqrt, pi, pow, exp #floor

#----------


def normal(x) :
    """ Normal
    """
    return np.exp(-np.square(x)/sqrt(2))/(sqrt(2*pi))


def gauss(x, x0=0, sig=1) :
    """ Gaussian
    """
    return normal((x-x0)/sig)/sig


def step(x) :
    return np.select([x>0,], [1,], default=0)


def step_smooth(x) :
    """ Smooth polinomial rising step from 0(x=0) to 1(x=1)
    """
    return np.select([x>1, x>0], [1, 3*np.square(x)-2*np.power(x,3)], default=0)


def step_linear_rise(x) :
    """ Rising step from 0(x=0) to 1(x=1)
    """
    return np.select([x>1, x>0], [1, x], default=0)


def step_sin_rise(x, width=1) :
    """ Sine-like rising step from 0(x<=0) to 1(x>=width)
    """
    a = (x-width/2)*pi/width
    return np.select([x>width, x>0], [1, (np.sin(a)+1)/2], default=0)


def step_sin_sharp_rise(x, width=1) :
    """ Sine-like rising step from 0(x<=0) to 1(x>=width), sharp kink at x=0
    """
    a = x*pi/(2*width)
    return np.select([x>width, x>0], [1, np.sin(a)], default=0)


def step_exp_rise(x, width=1) :
    """ exp-like rising step from 0(x<=0) to 1(x->inf)
    """
    return np.select([x>0,], [1-np.exp(-x/width),], default=0)


def step_gauss_rise(x, x0=0, sig=1) :
    """ exp-like rising step from 0(x<=0) to 1(x->inf)
    """
    return np.cumsum(gauss(x, x0, sig))


def linear_rise(x, x0=1) :
    """ Triangle rise from 0(x<=0) to 1(x=x0) then 0(x>x0)
    """
    return np.select([x>x0, x>0], [0, x/x0], default=0)


def exp_desc_step(x) :
    return np.select([x>0,], [np.exp(-x),], default=0)

#----------
#----------

if __name__ == "__main__" :

  import sys
  from time import time
  import psana.pyalgos.generic.Graphics as gr
  import matplotlib.pyplot as plt

  #logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s: %(message)s',\
  #                    datefmt='%m-%d-%Y %H:%M:%S',\
  #                    level=logging.DEBUG)

  def test_plot(fun, x=np.arange(-2, 2, 0.1), **kwargs) :
    fig = plt.figure(figsize=(12,5), dpi=80, facecolor='w', edgecolor='w', frameon=True, **kwargs)
    fig.canvas.set_window_title('Waveform dwt decomposition', **kwargs)
    ax = fig.add_axes((0.05, 0.05, 0.87, 0.9), **kwargs)
    ax.set_xlim((x[0], x[-1]))
    y = fun(x)
    ax.plot(x, y, 'b-', linewidth=1, **kwargs)
    plt.show()

  #----------

  tname = sys.argv[1] if len(sys.argv) > 1 else '1'
  print(50*'_', '\nTest %s' % tname)
  t0_sec=time()
  if   tname == '1': test_plot(normal)
  elif tname == '2': test_plot(gauss)
  elif tname == '3': test_plot(step_smooth)
  elif tname == '4': test_plot(step_linear_rise)
  elif tname == '5': test_plot(step_sin_rise)
  elif tname == '6': test_plot(step_sin_sharp_rise)
  elif tname == '7': test_plot(step_exp_rise)
  elif tname == '8': test_plot(step_gauss_rise)
  elif tname == '9': test_plot(linear_rise)
  elif tname =='10': test_plot(exp_desc_step)
  else : usage(); sys.exit('Test %s is not implemented' % tname)
  msg = 'Test %s consumed time %.3f' % (tname, time()-t0_sec)
  sys.exit(msg)

#----------

