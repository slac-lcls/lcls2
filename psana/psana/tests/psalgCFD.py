import constFracDiscrim as cfd
import numpy as np
import math

num_samples = 10000
sample_interval = math.pi
horpos = 0
gain = 10
offset = 0

times = np.linspace(0, math.pi, num=num_samples)
waveform = gain*np.sin(times)

delay = 1
walk = 40
threshold = 8
fraction = 0.5

peak_time = cfd.cfd(sample_interval, horpos, gain, offset, waveform, delay, walk, threshold, fraction)
assert(abs(peak_time - 1.58732246) < 1e-8)
