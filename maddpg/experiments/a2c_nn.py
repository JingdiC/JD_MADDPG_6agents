"""
Actor-Critic using TD-error as the Advantage, Reinforcement Learning.

The cart pole example. Policy is oscillated.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
tensorflow 1.0
gym 0.8.0
Mofan, pl for actor, td for critic
"""

import numpy as np
import tensorflow as tf
import gym
from mpe.multiagent.scenarios.simple_spread_way2 import Scenario
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(100)
tf.set_random_seed(100)  # reproducible

# Superparameters
OUTPUT_GRAPH = False
MAX_EPISODE = 1
DISPLAY_REWARD_THRESHOLD = 200  # renders environment if total episode reward is greater then this threshold
MAX_EP_STEPS = 30000 # maximum time step in one episode
RENDER = False  # rendering wastes time
GAMMA = 0.9  # reward discount in TD error
LR_A = 0.001  # learning rate for actor
LR_C = 0.01  # learning rate for critic

# env = World(2, GAMMA)

num_agents = 2

N_F = num_agents
N_A = num_agents



class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.a = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error

        with tf.variable_scope('Actor'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=250,  # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.acts_prob = tf.layers.dense(
                inputs=l1,
                units=n_actions,  # output units
                activation=tf.nn.softmax,  # get action probabilities
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='acts_prob'
            )

        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.acts_prob[0, self.a])
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss/ actor loss

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)

    def learn(self, s, a, td_error):
        s = s[np.newaxis, :]  # make it as row vector by inserting an axis along first dimension (1, n_state)
        feed_dict = {self.s: s, self.a: a, self.td_error: td_error}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})  # get probabilities for all actions
        # print(probs)
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())  # return a int


class Critic(object):
    def __init__(self, env, sess, n_features, lr=0.01):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.td_r = tf.placeholder(tf.float32, None, 'td_r')

        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=250,  # number of hidden units
                activation=tf.nn.relu,  # None
                # have to be linear to make sure the convergence of actor.
                # But linear approximator seems hardly learns the correct Q.
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.v = tf.layers.dense(
                inputs=l1,
                units=1,  # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='V'
            )

        with tf.variable_scope('squared_td_error'):
            combine_rewards = 0
            for i, agent in enumerate(env.agents):
                combine_rewards += self.td_r[i]
            self.td_error = combine_rewards + GAMMA * self.v_ - self.v
            self.loss = tf.square(self.td_error)  # TD_error = (r+gamma*V_next) - V_eval = target - prediction
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, td_r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _ = self.sess.run([self.td_r, self.train_op],
                                    {self.s: s, self.v_: v_, self.td_r: td_r})
        return td_error


sess = tf.Session()

actor = Actor(sess, n_features=N_F, n_actions=N_A, lr=LR_A)


critic = Critic(env, sess, n_features=N_F, lr=LR_C)



# we need a good teacher, so the teacher should learn faster than the actor

sess.run(tf.global_variables_initializer())

if OUTPUT_GRAPH:
    tf.summary.FileWriter("logs/", sess.graph)

best_selection_rates = []
for i_episode in range(MAX_EPISODE):
    s = env.reset()
    t = 0
    jump = 0
    track_r = []
    best_selections = []
    R_1, R_2, logR1, logR2, logRs = [], [], [], [], [] ## average rewards and log (r/average_reward)
    while True:

        a = actor.choose_action(s)
        s_, r, done, best_selection = env.step(a)
        t += 1
        ## logR for this step
        sumR = 0.0
        sum_logR = 0.0
        for i, agent in enumerate(env.agents):
            R = agent.average_step_reward  ## total reward / steps
            logR = np.log(R)
            if i == 0:
                R_1.append(R)
                logR1.append(logR)
            if i == 1:
                R_2.append(R)
                logR2.append(logR)

            sumR += R
            # sumR += r[i] / R  ## r_1/R_1 + r_2/R_2
        sum_logR = logR1[-1] + logR2[-1]
        logRs.append(sum_logR)

        best_selections.append(best_selection)

        td_error = critic.learn(s, r, s_)

        actor.learn(s, a, td_error)  # true_gradient = grad[logPi(s,a) * td_error]

        s = s_

        best_selection_rate = float(sum(best_selections) / len(best_selections))
        if done or t >= MAX_EP_STEPS:
            # print("logRs_compare", logRs[-1]/-0.43011641, " Ri_compare: ", R_1[-1]/0.65849 + R_2[-1]/0.98775)
            print("logRs_compare", logRs[-1]/1.2029347599724918, " Ri_compare: ", R_1[-1]/1.5669999459630626 + R_2[-1]/2.125000061001427)

            break

    best_selection_rates.append(best_selection_rate)

    R_1_csv = pd.DataFrame(R_1).to_csv("R_1_approach_2_2USERS_no_alpha.csv", index= False)
    R_2_csv = pd.DataFrame(R_2).to_csv("R_2_approach_2_2USERS_no_alpha.csv", index=False)
    logR_csv = pd.DataFrame(logRs).to_csv("~/Desktop/logRs_approach_2_2USERS_no_alpha.csv", index=False)

plt.plot(R_1, label = 'R_1')
plt.plot(R_2, label = 'R_2')
plt.legend(loc = 'upper right')
plt.xlabel('Steps')
plt.ylabel('R_1 and R_2')
plt.show()

plt.plot(logRs, label = 'logR1+logR2')
plt.plot(logR1, label = 'logR1')
plt.plot(logR2, label = 'logR2')
plt.legend(loc = 'upper right')
plt.xlabel('Steps')
plt.ylabel('logR_1 + logR_2, logR_1, logR_2')
plt.show()

# R_1[-1]/0.65849 + R_2[-1]/0.98775 = 1.99