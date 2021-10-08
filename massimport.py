# Data Visualization
# from som import SOM - GPU Only
#import ppaquette_gym_doom
#import roboschool
#import tradesys
#import autokeras as ak
#from auto_ml import Predictor
import cv2
import functools
import gc
import gym
#import gym_ple
#import keras
#import keras.backend as K
import matplotlib.pyplot as plt
import matplotlib.pyplot as pyplot
import mdp
import MultiNEAT as NEAT
import networkx
import networkx as nx
import numpy as np
import operator
import os
import pandas
import pandas as pd
import pickle
import progressbar as pb
import progressbar as pbar
import random as rnd
import seaborn as sns
import sklearn.model_selection
from sklearn.model_selection import StratifiedKFold
import sys
#import tensorflow as tf
#import theano
import time
import uuid
pbar = pb
#from modules.Anfis import Anfis
from modules.autoencoder import AutoEncoder
#from autokeras.classifier import *
# from modules.bayesianInference import BayesianInference
# from modules.bayes_clf import BayesClf
from modules.CatBoostWrapper import CatBoostClassifierWrapper
from catboost import CatBoostRegressor
from catboost import CatBoostClassifier
from collections import OrderedDict
from modules.custom_models_timeseries import *
from datacleaner import autoclean
from modules.defragTreesClassifer import DefragTreesClassifier
from modules.elm import ELM
#from evolutionary_search import EvolutionaryAlgorithmSearchCV
from modules.fcn import FCN
from modules.feature_learning import FeatureLearning
from modules.forward_NN_with_back_prop import FowardNNWithBackProp
from modules.forward_thinking_nn import ForwardNN
from functools import reduce
from modules.fuzzy import FuzzyClustering
from modules.fuzzyart import AdaptiveResonanceModel
from sklearn.gaussian_process.kernels import RBF
#newly added
from modules.rotation_adaboost import RotationAdaBoostClassifier
from modules.rotation_forest import RotationForestClassifier
#from modules.rotation_forest2 import RotationForestClassifier2
from modules.rotation_forest3 import RotationForest
from sklearn.semi_supervised import LabelPropagation
from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import RadiusNeighborsClassifier
from modules.vadaboost import BinaryVadaboost
from modules.logiboost import BinaryLogitBoost
from modules.class_switching import ClassSwitching
from modules.arc_x4 import ArcX4
from modules.fast_fm import FastFM
from modules.bernoulliRBM  import SKBernoulliRBM
from modules.gentleboost import GentleBoostClassifier
from modules.logit_boost import  LogitBoostClassifier
#from modules.lbarn import lBARN
from modules.brown_boost import BrownBoost
from modules.milboost import MILBoostClassifier


#newly added

#from gafe import GAFE
#from gafe.new_feature_set import NewFeatureSet
from modules.gam import GAMCLF
from modules.GCForest import gcForest
#from modules.Gmdhpy import GMDH
from modules.gmm import GMM
from gplearn.genetic import SymbolicTransformer, SymbolicRegressor
from gym.wrappers import Monitor
from modules.infer_data_type import *
from modules.infinite_boosting import InfiniteBoostingWithHoldoutCLF,ResearchGradientBoostingBaseCLF,InfiniteBoostingCLF
from io import BytesIO
from ipyparallel import Client
"""
from keras.callbacks import ModelCheckpoint, EarlyStopping, TerminateOnNaN
from keras.initializers import he_normal, he_uniform
from keras.initializers import orthogonal, identity, lecun_normal, lecun_uniform, glorot_normal, glorot_uniform
from keras.initializers import zeros, ones, constant, random_normal, random_uniform, truncated_normal
from keras.layers import Input, Embedding, LSTM, Dense, merge
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU, ThresholdedReLU
from keras.layers.convolutional import Conv1D, ZeroPadding1D, ZeroPadding2D, ZeroPadding3D
from keras.layers.convolutional import Conv2D, Conv2DTranspose, SeparableConv2D
from keras.layers.core import ActivityRegularization
from keras.layers.core import RepeatVector, Reshape, Permute, Dense, Dropout, Activation, Flatten, Lambda
from keras.layers.local import LocallyConnected1D, LocallyConnected2D
from keras.layers.merge import concatenate, add, maximum, average, multiply
from keras.layers.noise import GaussianNoise, GaussianDropout
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling1D, AveragePooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import SGD, Adadelta, Adamax, Adagrad, Adam, RMSprop, Nadam
from keras.utils import to_categorical
from modules.keras_lstm import LSTM_TMS
from modules.keras_nn import Keras_NN
"""
from modules.kmeans import KmeansClustering
from modules.LM import LM
from modules.MGCForest import MGCForest, MultiGrainedScanner, CascadeForest
from mlens.ensemble import *
from mlens.ensemble import SuperLearner
from mlens.metrics import *
from mlens.model_selection import Evaluator
from modules.mlp import MLP
from MultiNEAT import viz
from MultiNEAT.viz import Draw
from modules.naive_bayes import NB
from OpenGL import GLU
from pomegranate import *
from modules.preprocessing import *
from modules.probabilistic import ProbabilisticClustering
from progressbar import ProgressBar, Counter, Timer, RotatingMarker, FileTransferSpeed, FormatCustomText
from random import randint
from modules.resnet import ResNet
#from modules.rgforest import RGForest
#import rgf
#from rl.agents.dqn import DQNAgent
#from rl.memory import SequentialMemory
#from rl.policy import BoltzmannQPolicy
#from rl.processors import Processor
#from modules.rr_extra_forest import RRExtraTreesClassifier
#from modules.rr_forest import RRForestClassifier
from scipy.stats import uniform, randint
from sklearn import *
from sklearn import model_selection as cross_validation
from sklearn import datasets
from sklearn import ensemble
from sklearn import feature_selection
#from sklearn import grid_search
from sklearn import linear_model
from sklearn import manifold
from sklearn import metrics
from sklearn import metrics, preprocessing, linear_model
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import svm
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from sklearn import tree as DecisionTreeClassifier
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import PCA, FastICA
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.dummy import *
from sklearn.ensemble import *
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.ensemble import VotingClassifier, BaggingClassifier
from sklearn.feature_selection import *
from sklearn.feature_selection import SelectPercentile, SelectFromModel, f_regression, SelectKBest, RFECV
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_approximation import *
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import *
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, LinearRegression, PassiveAggressiveClassifier, RidgeClassifierCV, Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.manifold import TSNE as tnse
from sklearn.manifold import TSNE, Isomap, SpectralEmbedding, MDS, LocallyLinearEmbedding
from sklearn.metrics import classification_report
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score, RandomizedSearchCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import *
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import *
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import *
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.semi_supervised import *
from sklearn.svm import *
from sklearn.svm import NuSVC
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import *
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from modules.snapshot_ensembles import SnapshotEnsembles, SnapshotCallbackBuilder, SnapshotModelCheckpoint
from modules.SNN import SNN
from modules.soinn import SOINN
from tensorflow.contrib import learn
from tpot import TPOTClassifier
from vecstack import *
from modules.wide_resnet import WideResNet
from xgboost import *
from xgboost import XGBClassifier

import lightgbm as lgb
import mlxtend
#from stepwise import StepWise
#from rgf import RGFClassifier
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import OrthogonalMatchingPursuitCV
from mlxtend.classifier import Adaline, Perceptron, StackingClassifier
from mlxtend.classifier import Perceptron as mlxtendPerceptron 
from mlxtend.classifier import LogisticRegression as mlxtendLogisticRegression
from mlxtend.classifier import MultiLayerPerceptron as mlxtendMLP

import time
import progressbar as pb
import uuid
pbar = pb
import random as rnd
"""
from keras.models import Sequential
from keras.layers import Input, Embedding, LSTM, Dense, merge
from keras.models import Model
from keras.layers.core import RepeatVector, Reshape, Permute, Dense, Dropout, Activation, Flatten
from keras.layers.core import ActivityRegularization
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU, ThresholdedReLU
from keras.layers.normalization import BatchNormalization
from keras.initializers import zeros, ones, constant, random_normal, random_uniform, truncated_normal
from keras.initializers import orthogonal, identity, lecun_normal, lecun_uniform, glorot_normal, glorot_uniform
from keras.initializers import he_normal, he_uniform
from keras.layers.convolutional import Conv2D, Conv2DTranspose, SeparableConv2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.layers.pooling import MaxPooling1D, AveragePooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers.convolutional import Conv1D, ZeroPadding1D, ZeroPadding2D, ZeroPadding3D
from keras.layers.local import LocallyConnected1D, LocallyConnected2D
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.optimizers import SGD, Adadelta, Adamax, Adagrad, Adam, RMSprop, Nadam
from keras.callbacks import ModelCheckpoint, EarlyStopping, TerminateOnNaN
from keras.layers.merge import concatenate, add, maximum, average, multiply
from keras.layers.noise import GaussianNoise, GaussianDropout
from keras.utils import to_categorical

import keras.backend as K
"""
import pandas as pd

from sklearn.preprocessing import LabelBinarizer, LabelEncoder, MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV

import MultiNEAT as NEAT
from MultiNEAT.viz import Draw
from MultiNEAT import viz
import networkx as nx
import networkx
from progressbar import ProgressBar, Counter, Timer, RotatingMarker, FileTransferSpeed, FormatCustomText
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.pyplot as pyplot
import operator
import random as rnd
import functools
#import tradesys
import uuid
import gc
import pickle




    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
