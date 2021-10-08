import numpy as np
import scipy as sp
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from collections import Counter
import time
import progressbar as pb
from tqdm import tqdm
import os
import shutil
from empyrical import sortino_ratio, calmar_ratio, omega_ratio, sharpe_ratio
from numba import jit


class TradingEnv(gym.Env):
    """ This gym implements a simple trading environment for reinforcement learning. """

    metadata = {'render.modes': ['human']}

    @jit
    def __init__(self, input_source, to_predict,
                 winlen=1, bars_per_episode=1000, traded_amt = 10000, initial_balance=10000,
                 commission = 0, slippage = 0,
                 reward_type = 'cur_balance', # 'balance', 'cur_balance', 'sortino'
                 max_position_time = 3,
                 min_ratio_trades = 8,
                 ):
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(  # np.min(input_source, axis=0),
            # np.max(input_source, axis=0)
            np.ones((winlen * input_source.shape[1] + 0,)) * -15,
            np.ones((winlen * input_source.shape[1] + 0,)) * 15,
        )
        self.input_source = input_source
        self.to_predict = to_predict
        self.winlen = winlen
        self.bars_per_episode = bars_per_episode
        self.traded_amt = traded_amt
        self.commission = commission
        self.slippage = slippage
        self.reward_type = reward_type
        self.initial_balance = initial_balance
        self.max_position_time = max_position_time
        self.min_ratio_trades = min_ratio_trades
        np.random.seed()
        self.reset()

    def _configure(self, display=None):
        self.display = display

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        return self.step(action)

    def _reset(self):
        return self.reset()

    @jit
    def step(self, action):

        # assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        if (self.idx < self.end_idx) and (self.balance > 0):
            self.idx += 1
            done = False
        else:
            done = True


        #try:
        #    if len(action)>1: action = int(np.argmax(action))
        #except:
        #    pass

        comm_paid = 2 * self.commission
        slip_paid = 2 * self.slippage * self.traded_amt

        ret = 0
        if self.position == -1:  # long
            ret = (self.to_predict[self.idx] - self.to_predict[self.open_idx]) * self.traded_amt - comm_paid - slip_paid
        elif self.position == 1:  # short
            ret = (self.to_predict[self.open_idx] - self.to_predict[self.idx]) * self.traded_amt - comm_paid - slip_paid


        # execute the action and get the reward
        if (action == 0) and (self.position == 0):  # buy
            self.position = -1
            self.open_idx = self.idx
        elif (action == 1) and (self.position == 0):  # sell
            self.position = 1
            self.open_idx = self.idx
        elif ((action == 2) and (self.position != 0))  or ((self.position!=0) and ((self.idx - self.open_idx) > self.max_position_time)): # close
            if self.position == -1:  # long
                self.balance += (self.to_predict[self.idx] - self.to_predict[
                    self.open_idx]) * self.traded_amt - comm_paid - slip_paid
                self.returns.append(ret)
            elif self.position == 1:  # short
                self.balance += (self.to_predict[self.open_idx] - self.to_predict[
                    self.idx]) * self.traded_amt - comm_paid - slip_paid
                self.returns.append(ret)
            self.position = 0
        elif action == 3:
            pass
        else:
            pass


        self.prev_balance = self.balance


        info = {}

        observation = np.hstack( [self.input_source[self.idx - self.winlen: self.idx, :].reshape(-1),
                                  #self.position
                                  ] )

        if self.reward_type == 'sortino':

            if len(self.returns) > self.min_ratio_trades:
                reward = sortino_ratio(np.array(self.returns[-self.min_ratio_trades:]))
                if np.isnan(reward) or np.isinf(reward):
                    reward = 0
            else:
                reward = 0
        elif self.reward_type == 'sharpe':

            if len(self.returns) > self.min_ratio_trades:
                reward = sharpe_ratio(np.array(self.returns[-self.min_ratio_trades:]))
                if np.isnan(reward) or np.isinf(reward):
                    reward = 0
            else:
                reward = 0
        elif self.reward_type == 'omega':

            if len(self.returns) > self.min_ratio_trades:
                reward = omega_ratio(np.array(self.returns[-self.min_ratio_trades:]))
                if np.isnan(reward) or np.isinf(reward):
                    reward = 0
            else:
                reward = 0
        elif self.reward_type == 'cur_balance':
            reward = ret
            if np.isnan(reward) or np.isinf(reward):
                reward = 0
        elif self.reward_type == 'balance':
            if len(self.returns) > 0:
                reward = np.sum(self.returns)#self.balance
            else:
                reward = 0
        elif self.reward_type == 'rel_balance':
            if len(self.returns) > self.min_ratio_trades:
                reward = np.sum(self.returns[-self.min_ratio_trades:])#self.balance
            else:
                reward = 0
        else:
            reward = 0

        #reward = reward * len(self.returns)

        return observation, reward, done, info

    @jit
    def reset(self):
        # reset and return first observation
        self.idx = np.random.randint(self.winlen, self.input_source.shape[0] - self.bars_per_episode)
        self.end_idx = self.idx + self.bars_per_episode
        self.position = 0
        self.open_idx = 0
        self.balance = self.initial_balance
        self.prev_balance = self.balance
        self.returns = []
        return np.hstack( [self.input_source[self.idx - self.winlen: self.idx, :].reshape(-1),
                           #self.position
                           ] )

    @jit
    def reset2(self):
        # reset and return first observation
        self.idx = self.winlen
        self.end_idx = self.idx + self.bars_per_episode
        self.position = 0
        self.open_idx = 0
        self.balance = self.initial_balance
        self.prev_balance = self.balance
        self.returns = []
        return np.hstack( [self.input_source[self.idx - self.winlen: self.idx, :].reshape(-1),
                          # self.position
                           ] )

    def _render(self, mode='human', close=False):
        # ... TODO
        pass
