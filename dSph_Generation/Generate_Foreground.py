import numpy as np
import pandas as pd
import scipy.stats.distributions as st
from scipy.integrate import quad, solve_ivp
from scipy.interpolate import interp1d
from scipy.stats import norm
from dSph_Model import GenerateForeground, chabrier_imf
import matplotlib.cm as cm
from matplotlib.ticker import LogLocator
import matplotlib.pyplot as plt
import time

fgnd = GenerateForeground()
fgnd.starCone()