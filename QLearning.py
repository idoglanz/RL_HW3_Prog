import numpy as np
from util import *


class QLearning:
    def __init__(self, env, alpha=1, batch_size=1, max_iterations=100, trajectories=100, epsilon=0, theta_size=9, epsilon_greedy_val=0, epsilon_greedy_val_flag=0):
        self.env = env
        self.alpha = alpha
        self.theta = np.random.rand(theta_size)

        self.batch_size = batch_size
        self.trajectories = trajectories
        self.max_iterations = max_iterations
        self.epsilon = epsilon

        self.epsilon_greedy_val = epsilon_greedy_val
        self.epsilon_greedy_val_flag = epsilon_greedy_val_flag

    def train(self, eval_by_n_starts=10):
        iteration = 1
        done = False
        data = DataSet(self.env, N=self.batch_size, trajectories=self.trajectories, method='Q_learning', iteration=iteration, epsilon_greedy_val=self.epsilon_greedy_val, epsilon_greedy_val_flag=self.epsilon_greedy_val_flag)
        simulation = PlaySimulation(self.env, data, starts=eval_by_n_starts)
        success_rate = []

        while not done:
            data.collect(theta=self.theta, iteration=iteration)
            data.normalize()

            # self.theta = self.theta + (self.alpha/(self.batch_size*self.trajectories)) * np.dot(data.reward  * np.dot(data.phi_next, self.theta) - np.dot(data.phi, self.theta), data.phi)
            self.theta = self.theta + (self.alpha/len(data.samples)) * np.dot(data.reward * np.dot(data.phi_next, self.theta) - np.dot(data.phi, self.theta), data.phi)

            success_rate = np.append(success_rate, simulation.success(self.theta))

            iteration += 1

            if iteration > self.max_iterations or success_rate[-1] > 90:
                done = True
                iteration_fill = iteration
                while iteration_fill <= self.max_iterations:
                    success_rate = np.append(success_rate, success_rate[-1])
                    iteration_fill += 1

        # simulation.play(self.theta)
        return self.theta, [success_rate[-1], iteration], success_rate
        # return self.theta, data

