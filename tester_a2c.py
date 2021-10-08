from numpy import *
from matplotlib.pyplot import *
import numpy as np
import matplotlib.pyplot as plt

import gym
from gym import error, spaces, utils
from gym.utils import seeding
from collections import Counter
import time
import progressbar as pb
from tqdm import tqdm
import os
import shutil
from empyrical import sortino_ratio, calmar_ratio, omega_ratio
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, VecEnv, VecEnvWrapper
from stable_baselines import A2C, PPO2
from stable_baselines.common.vec_env import DummyVecEnv

from trading_env import TradingEnv

# load the market data
input_source = np.load(open('data_eurusd.npy','rb'))
to_predict = np.load(open('data_eurusd_targets.npy','rb'))

to_predict = to_predict[3,:].reshape(-1)

input_source = input_source.T

is_orig = np.copy(input_source)
cp = int(0.8*len(input_source))
test_input_source = input_source[cp:, :]
test_to_predict = to_predict[cp:]
input_source = input_source[0:cp, :]
to_predict = to_predict[0:cp]

bars_per_episode = 1000
winlen = 1
traded_amt = 10000
commission = 0
slippage = 0.0




# multiprocess environment
#n_cpu = 16
#env = SubprocVecEnv([lambda: TradingEnv() for i in range(n_cpu)])
env = TradingEnv(input_source, to_predict,
                 winlen=winlen, bars_per_episode=bars_per_episode, traded_amt=traded_amt,
                 commission=commission, slippage=slippage,
                 reward_type='sortino'
                 )
env = DummyVecEnv([lambda: env])

t = 0
[shutil.rmtree('/home/peter/tblog/'+x) for x in os.listdir('/home/peter/tblog/') if x]
model = A2C(MlpLstmPolicy, env, verbose=1, tensorboard_log='/home/peter/tblog')
try:
    model.learn(total_timesteps=1_000_000)
    model.save("a2c_trading")
except KeyboardInterrupt:
    pass


env = TradingEnv(test_input_source, test_to_predict,
                 winlen=winlen, bars_per_episode=bars_per_episode, traded_amt=traded_amt,
                 commission=commission, slippage=slippage
                 )
env = DummyVecEnv([lambda: env])


# visualize the behavior for one random episode
bars_per_episode = 1000

nstate = model.initial_state  # get the initial state vector for the reccurent network
#dones = np.zeros(nstate.shape[0])  # set all environment to not done
nstate=None

observation = env.envs[0].reset()#env.reset()
done = False
navs = []
acts = []
for i in tqdm(range(bars_per_episode)):
    action, nstate = model.predict([observation], state=nstate)
    acts.append(action)
    observation, reward, done, info = env.envs[0].step(action)#env.step(action)
    if done:
        break
    navs.append(env.get_attr('balance')[0])


kl = []
t = 0
for n in np.diff(np.vstack(navs).reshape(-1)):
    t = t + n
    kl.append(t)
plot(kl);

# calculate the likelihood of success for any given episode
try:
    l = 250

    krl = []
    p = pb.ProgressBar(max_value=l)
    for i in range(l):
        p.update(i)
        observation = env.envs[0].reset()
        done = False
        navs = []
        for i in (range(bars_per_episode)):
            action, nstate = model.predict([observation], state=nstate)
            acts.append(action)
            observation, reward, done, info = env.envs[0].step(action)#env.step(action)
            navs.append(env.get_attr('balance')[0])
        krl.append(sum(navs))
    p.finish()
except KeyboardInterrupt:
    pass

krl = np.array(krl)
print('Profit likelihood: %3.3f%%' % (100*(sum(krl > 0) / len(krl))))

hist(krl);

