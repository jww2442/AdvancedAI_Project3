from CartPole_v0_using_Q_learning_SARSA_and_DNN.Qlearning_for_cartpole import CartPoleQAgent
from CartPole_v0_using_Q_learning_SARSA_and_DNN.Sarsa_for_cartpole import CartPoleAgent
from cartpole_balancing.tabular_solutions import cartpole, cartpole_env

import shutil
import matplotlib.pyplot as plt
import numpy as np

NUM_TRIALS = 100

def main():
    Q_t = []
    S_t = []
    VP_t = []

    cartpole_g = cartpole_env.CartPoleEnv()

    for i in range(NUM_TRIALS):
        Qagent = CartPoleQAgent()
        Qr = Qagent.train()
        t = Qagent.run()
        Q_t.append(t)
        '''manifest_dir = 'CartPole_v0_using_Q_learning_SARSA_and_DNN/cartpole'
        try:
            shutil.rmtree(manifest_dir)
        except OSError as e:
            print("Error: %s : %s" % (manifest_dir, e.strerror))
        '''

        Sagent = CartPoleAgent()
        Sr = Sagent.train()
        t = Sagent.run()
        S_t.append(t)
        '''manifest_dir = 'CartPole_v0_using_Q_learning_SARSA_and_DNN/cartpole'
        try:
            shutil.rmtree(manifest_dir)
        except OSError as e:
            print("Error: %s : %s" % (manifest_dir, e.strerror))
        '''

    t, e, r = cartpole.run(cartpole_g)
    VP_t.append(t/(e*10))
    VP_tt = np.repeat(VP_t, 100)
    VPr = r

    plot_res(Q_t, S_t, VP_tt, Qr, Sr, VPr)

def plot_res(Q_t, S_t, VP_t, Qr, Sr, VPr):
    plt.figure()
    plt.xlabel("Trial")
    plt.ylabel("Time Upright")
    title_str = 'Time Upright After Training, Averaged Across ' + str(NUM_TRIALS) + ' Trials'

    trials = list(range(1, NUM_TRIALS+1))
    Q_trial_avg_t =[get_avg(Q_t) for i in range(NUM_TRIALS)]
    S_trial_avg_t =[get_avg(S_t) for i in range(NUM_TRIALS)]
    VP_trial_avg_t =[get_avg(VP_t) for i in range(NUM_TRIALS)]

    plt.plot(trials, Q_t, label = "Actual Time Upright for Q-Learning", color='black')
    plt.plot(trials, Q_trial_avg_t, label = "Average Time Upright Across Trials for Q-Learning", color = 'black', linestyle='dashed')
   
    plt.plot(trials, S_t, label = "Actual Time Upright for SARSA", color='blue', alpha=0.35)
    plt.plot(trials, S_trial_avg_t, label = "Average Time Upright Across Trials for SARSA", color = 'blue', linestyle='dashed')
   
    plt.plot(trials, VP_t, label = "Actual Time Upright for Policy Iteration", color='red')
    plt.plot(trials, VP_trial_avg_t, label = "Average Time Upright Across Trials for Policy Iteration", color = 'red', linestyle='dashed')
   
    plt.title(title_str)
    plt.legend()
    plt.show(block=False)
    plt.savefig('Figures/Cartpole/1.png')

    plt.figure()
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    eps = list(range(1, 1001))
    plt.plot(eps, Qr, label = "Q-learning rewards", color='black')
    plt.plot(eps, Sr, label = "SARSA rewards", color='blue', alpha=0.35)
    #plt.plot(eps, VPr, label = "Policy Iteration rewards", color='red')
    plt.title("Rewards per Episode during Training")
    plt.legend()
    plt.show(block=False)
    plt.savefig('Figures/Cartpole/2.png')

    

def get_avg(myList):
    return (sum(myList)/len(myList))

if __name__ == '__main__':
    main()

