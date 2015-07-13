from pybrain.supervised.trainers.backprop import BackpropTrainer
from pybrain.structure import RecurrentNetwork, LinearLayer, SigmoidLayer ,TanhLayer, BiasUnit, FullConnection, IdentityConnection
from pybrain.datasets import SupervisedDataSet 
import matplotlib.pyplot as plt
import numpy as np 
from random import random as rnd