# Project 3 for the University of Tulsa's CS-7313 Adv. AI Course
# Solving Multi-Armed Bandit Problems
# Professor: Dr. Sen, Fall 2021
# Noah Schrick, Noah Hall, Jordan White

#from Bandit_Sim import bandit_sim as bsim

from numpy import log10, sqrt
import random
from Bandit_Sim.bandit_sim import Bandit_Sim


NUM_ARMS = 3

class Arm:
    instances = []
    def __init__(self):
        self.emp_mean = 0
        self.avg = 0
        self.pulls = 0
        Arm.instances.append(self)
    
    def update_avg(self, value):
        self.avg = ((self.pulls * self.avg + value)/ self.pulls +1)

def main():
    #Init Bandit Simulator
    bandsim = Bandit_Sim(NUM_ARMS, 0.1)
    #Initialize Arms
    for i in range(NUM_ARMS):
        i = Arm()
    #Call UCB
    V = ucb(100, NUM_ARMS, bandsim)
    #bandsim.plot(100)
    print(bandsim.arm_means)
    print(V)
    egreedy(100, NUM_ARMS, 0.1, bandsim)

def ucb(samples, num_arms, bandsim):
    V = 0
    #Try all actions before the loop:
    for arm in range(num_arms):
        r = bandsim.pull_arm(arm)
        V += r
        Arm.instances[arm].emp_mean = (r/2)
        Arm.instances[arm].pulls += 1
    #Repeat Loop:
    for i in range(1, samples):
        #Argmax
        curr_max = 0
        max_idx = None
        for arm in range(num_arms):
            arg_val = Arm.instances[arm].emp_mean + sqrt((2 * log10(i))/Arm.instances[arm].pulls)
            if arg_val > curr_max:
                curr_max = arg_val
                max_idx = arm
        
        #Pull arm with highest Hoeffding bound
        r = bandsim.pull_arm(max_idx)
        V += r
        na = Arm.instances[max_idx].pulls
        Arm.instances[max_idx].emp_mean = ((na * Arm.instances[max_idx].emp_mean + r)/(na +1))
        Arm.instances[max_idx].pulls += 1
    
    return V

def egreedy(samples, num_arms, e, bandsim):
    tmp = 1
    for i in range(samples):
        rand_num = random.random()
        #Randomly choose an arm if random number meets criteria 
        if rand_num > (1-e):
            max_idx = random.choice(range(len(Arm.instances)))
        #Else pick arm with highest average so far:
        else:
            max_idx = max(range(len(Arm.instances)), key=lambda i: Arm.instances[i].avg)
        
        #Get reward
        payout = bandsim.pull_arm(max_idx)
        #Update average
        Arm.instances[max_idx].update_avg(payout)


if __name__ == '__main__':
    main()

