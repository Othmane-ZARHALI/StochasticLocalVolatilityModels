""" Project : SLV calibration
Done by : Othmane ZARHALI

"""

# Packages
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import scipy.interpolate as inter
import scipy.stats
from scipy.stats import norm
from scipy.special import gamma, factorial2
from math import *
from random import gauss
from scipy import linalg

# BERGOMI SLV MODEL #####################
# Model Parameters :
theta=0.4
rhoXY=0.01
kappaX=0.7
kappaY=0.7
sigma = 0.3
xi0 = 0.03
model_parameter_list = [theta,rhoXY,kappaX,kappaY,sigma]


# Kernel fuction
def gauss_integral(n):
    r"""
    Solve the integral
    \int_0^1 exp(-0.5 * x * x) x^n dx
    See
    https://en.wikipedia.org/wiki/List_of_integrals_of_Gaussian_functions
    Examples
    --------
    #>>> ans = gauss_integral(3)
    #>>> np.allclose(ans, 2)
    True
    #>>> ans = gauss_integral(4)
    #>>> np.allclose(ans, 3.75994)
    True
    """
    factor = np.sqrt(np.pi * 2)
    if n % 2 == 0:
        return factor * factorial2(n - 1) / 2
    elif n % 2 == 1:
        return factor * norm.pdf(0) * factorial2(n - 1)
    else:
        raise ValueError("n must be odd or even.")

def gaussian(x, dims=1):
    normalization = dims * gauss_integral(dims - 1)
    dist_sq = x ** 2
    return np.exp(-dist_sq / 2) / normalization

def forward_Variance(t,T,the):

    X = (1-exp(-2*kappaX*T))/(2*kappaX)*gauss(0,1)
    Y = (1 - exp(-2 * kappaY * T)) / (2 * kappaY) * gauss(0,1)
    alpha =  ((1 - the) ** 2 + the ** 2 + 2 * rhoXY * the * (1 - the)) ** (-1 / 2)
    x = alpha * ((1 - the) * exp(-kappaX * (T - t)) * X + the * exp(-kappaY * (T - t)) * Y)
    h = (1 - theta) ** 2 * exp(-2 * kappaX * (T - t)) * (
                (1 - exp(-2 * kappaX * T)) / (2 * kappaX)) + theta ** 2 * exp(-2 * kappaY * (T - t)) * (
                                 (1 - exp(-2 * kappaY * T)) / (2 * kappaY)) + 2 * theta * (1 - theta) * exp(
        -(kappaX + kappaY) * (T - t)) * ((1 - exp(-(kappaX + kappaY) * T)) / (kappaX + kappaY))
    f =  exp(2 * sigma * x- 2 * sigma ** 2 * h)
    return xi0 * f


# Bergomi simulator
def Bergomi_simulator(maturity,localvolatilitylist,N_sample,tk,tkplusone,initialassetlist):
    '''It returns the simulation of the the couple (S,V) '''

    Volatility_simulation = np.ones(N_sample)
    time_step = tkplusone-tk
    G = np.random.normal(np.zeros(N_sample), np.ones(N_sample))
    for i in range(N_sample):
        Volatility_simulation[i]=forward_Variance(tk,maturity,theta)
    Asset_simulation = np.array(initialassetlist)  + np.array(initialassetlist)*sqrt(time_step) * localvolatilitylist*np.sqrt(np.array(forward_Variance(tk,maturity,theta)))*G

    return Asset_simulation,Volatility_simulation

def SLV_calibration(maturity,Dupirelocalvolfunction,N_sample,N_timedisc,S0):
    assetlist = S0*np.ones(N_sample)
    localvolatility_fun =  lambda K:Dupirelocalvolfunction(0,K)/sqrt(xi0)
    localvolatility_list =[localvolatility_fun(x) for x in assetlist]
    time_discretisation = np.linspace(0, maturity, N_timedisc)
    localvolatilityfunctionlist = [localvolatility_fun]  #a list per time discretisation point
    kernelbandwidth = N_sample**(-1/5)

    for i in range(1,N_timedisc):
        bergomi_simulator = Bergomi_simulator(maturity,localvolatility_list,N_sample,time_discretisation[i-1],time_discretisation[i],assetlist)
        asset_simulator = bergomi_simulator[0]
        volatility_simulator = bergomi_simulator[1]
        # kernel_list_evaluation
        current_local_volatility_function = lambda K:Dupirelocalvolfunction(time_discretisation[i],K)*\
                                                     sqrt(sum([gaussian((x-K)/kernelbandwidth) for x in asset_simulator])/sum(volatility_simulator**2*gaussian((asset_simulator-K)/kernelbandwidth)))

        localvolatilityfunctionlist.append(current_local_volatility_function)
        localvolatility_list = [current_local_volatility_function(x) for x in assetlist]

    return localvolatilityfunctionlist


Dupirelocalvolfunction =lambda t,x:0.05
print("test =",SLV_calibration(5,Dupirelocalvolfunction,10,10,10))

