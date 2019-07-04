# import numpy as np
import mountain_car_with_data_collection as sim
# import time
from LSPI import *
from QLearning import *
from PolicyGradient import *
from util import *


def q3_testing_the_model(env):
    print('Q3 - testing the model')

    for N in [10000, 5000, 1000, 20000]:
    # for N in [10000]:

        method = 'LSPI'
        evaluation = 5
        success_rate_final = np.zeros((evaluation, 2))
        success_rate = list()
        max_iterations = 20

        for i in range(evaluation):
            model = LSPIModel(env, gamma=0.999, max_iterations=max_iterations, epsilon=0, theta_size=15)
            data = DataSet(env, N=N, method=method)
            data.collect()
            data.normalize()
            simulation = PlaySimulation(env, data=data, starts=10)

            theta, success_rate_final[i], success_rate_temp = model.train(data, simulation)

            success_rate.append(success_rate_temp)
        plot_success(success_rate, max_iteration=max_iterations, average=evaluation, N=N)
        print('success rate: ', success_rate_final[:, 0], 'iterations', success_rate_final[:, 1])
    plt.show()


def q4_testing_the_model(env):
    print('Q4 - testing the model')
    evaluation = 5
    eval_by_n_starts = 10
    success_rate_final = np.zeros((evaluation, 2))
    success_rate = list()
    max_iterations = 50

    for i in range(evaluation):
        model = QLearning(env, alpha=0.1, batch_size=1, trajectories=100, max_iterations=max_iterations, epsilon=1e-4, theta_size=15, epsilon_greedy_val=0.1)
        # theta, data = model.train()
        theta, success_rate_final[i], success_rate_temp = model.train(eval_by_n_starts=eval_by_n_starts)
        success_rate.append(success_rate_temp)

    plot_success(success_rate, max_iteration=max_iterations, average=evaluation, N=10)
    # print('success rate: ', success_rate_final[:, 0], 'iterations', success_rate_final[:, 1])

    plt.show()


#######################################################################################################################


if __name__ == '__main__':
    env = sim.MountainCarWithResetEnv()

    q3_testing_the_model(env)
    # q4_testing_the_model(env)

    PolicyGradient(env)

#######################################################################################################################
