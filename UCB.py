# Project 3 for the University of Tulsa's CS-7313 Adv. AI Course
# Solving Multi-Armed Bandit Problems
# Professor: Dr. Sen, Fall 2021
# Noah Schrick, Noah Hall, Jordan White

# For writing Figures to Report.docx
#from docx import Document
#from docx.shared import Inches

import numpy as np
import random
from Bandit_Sim.bandit_sim import Bandit_Sim
import matplotlib.pyplot as plt

#Global Variables
NUM_ARMS = 10
NUM_SAMPLES = 1000
EPSILON = 0.3
NUM_TRIALS = 100
STD_DEV = 0.1

class Arm:
    instances = []
    def __init__(self):
        self.emp_mean = 0
        self.avg = 0
        self.pulls = 0
        self.total_pulls = 0
        Arm.instances.append(self)
    
    def update_avg(self, value):
        self.avg = ((self.pulls * self.avg + value)/ (self.pulls +1))

    def pull_reset(self):
        self.total_pulls = self.pulls
        self.pulls = 0

def main():
    #Init Bandit Simulator
    bandsim = Bandit_Sim(NUM_ARMS, STD_DEV)
    #Initialize Arms
    for i in range(NUM_ARMS):
        i = Arm()
    
    #Init trial lists
    ucb_tot_r_list = []
    ucb_regret = []
    eg_tot_r_list = []
    eg_regret = []

    # Writing to Document
    #document = Document('Report.docx')
    #heading_str = str(NUM_SAMPLES) + ' Samples, ' + str(NUM_TRIALS) + ' Trials, ' + str(NUM_ARMS) + ' Arms, Epsilon=' + str(EPSILON) + ' Std Dev=' + str(STD_DEV)
    #document.add_heading(heading_str)
    #document.save('Report.docx')

    '''Call UCB'''
    for i in range(NUM_TRIALS):
        V, ucb_r_list = ucb(NUM_SAMPLES, NUM_ARMS, bandsim)
        ucb_tot_r_list.append(ucb_r_list)
        ucb_regret.append(get_regret(V, NUM_SAMPLES, bandsim))
        for i in range(NUM_ARMS):
            Arm.instances[i].pull_reset()

    ucb_r_avg = list_avg(ucb_tot_r_list)
    plot_results(ucb_r_avg, ucb_regret, "UCB", bandsim)

    '''Reset and Call e-greedy'''
    for i in range(NUM_ARMS):
        Arm.instances[i].pull_reset()

    for i in range(NUM_TRIALS):    
        V, eg_r_list = egreedy(NUM_SAMPLES, NUM_ARMS, EPSILON, bandsim)
        eg_tot_r_list.append(eg_r_list)
        eg_regret.append(get_regret(V, NUM_SAMPLES, bandsim))
        for i in range(NUM_ARMS):
            Arm.instances[i].pull_reset()
    
    eg_r_avg = list_avg(eg_tot_r_list)
    plot_results(eg_r_avg, eg_regret, "e-Greedy", bandsim)

    comp_ucb_eg(ucb_regret, eg_regret, bandsim)
    plt.show()

  
def plot_results(r_list, regret, fcn, bandsim):
    #Plot Convergence
    plt.figure()
    plt.plot((list(range(1, NUM_SAMPLES))), r_list, label = "Actual Reward")
    plt.xlabel("Sample")
    plt.ylabel("Payout")
    if fcn == "UCB":
        title_str = 'Reward over time for ' + fcn + ', Averaged Across ' + str(NUM_TRIALS) + ' Trials\n' + str(NUM_SAMPLES) + ' Samples'
    else:
        title_str = 'Reward over time for ' + fcn + ', Averaged Across ' + str(NUM_TRIALS) + ' Trials\n' + str(NUM_SAMPLES) + ' Samples, Epsilon=' + str(EPSILON)

    for arm in range(NUM_ARMS):
        rew = bandsim.arm_means[arm]
        labelstr = 'Action ' + str(arm) + ' Means Reward'
        plt.plot(list(range(1, NUM_SAMPLES)), [rew for i in range(1, NUM_SAMPLES)], label = labelstr)
        plt.text(NUM_SAMPLES, rew, str(rew))
    plt.title(title_str)
    plt.legend()
    plt.show(block=False)
    plt.savefig('1.png')
    #document=Document('Report.docx')
    #document.add_picture('1.png', width=Inches(1.25))

    #Bar Graph:
    #Get list of arms for plotting labels
    arms = list(range(1, NUM_ARMS+1))
    for i in arms:
        arms[i-1] = str(i)

    #Get total number of pulls for each arm
    tot_pulls = list(Arm.instances[i].total_pulls/NUM_TRIALS for i in range(NUM_ARMS))
    plt.figure()
    plt.bar(arms, tot_pulls)
    plt.xlabel("Arm")
    ystr_txt = 'Number of Pulls'  
    plt.ylabel(ystr_txt)
    if fcn == "UCB":
        title_str = 'Number of Pulls of each Action for ' + fcn + ', Averaged Across ' + str(NUM_TRIALS) + ' Trials\n' + str(NUM_SAMPLES) + ' Samples'
    else:
        title_str = 'Number of Pulls of each Action for ' + fcn + ', Averaged Across ' + str(NUM_TRIALS) + ' Trials\n' + str(NUM_SAMPLES) + ' Samples, Epsilon=' + str(EPSILON)
        
    plt.title(title_str)
    #Show values on top of each bar
    for i in range(len(arms)):
        plt.text(i, tot_pulls[i], tot_pulls[i], ha="center", va="bottom")
    #plt.show(block=False)
    plt.savefig('2.png')
    #document.add_picture('2.png', width=Inches(1.25))


    #Plot Regret
    trials = list(range(1, NUM_TRIALS+1))
    if fcn == "UCB":
        theo_regret = [np.log10(NUM_SAMPLES) for i in range(NUM_TRIALS)]
    else:
        theo_regret = [EPSILON * NUM_SAMPLES * ((NUM_ARMS -1)/NUM_ARMS) for i in range(NUM_TRIALS)]
    trial_avg_regret =[get_avg(regret) for i in range(NUM_TRIALS)]
    plt.figure()
    plt.plot(trials, regret, label = "Actual Means Regret")
    #plt.plot(trials, theo_regret, label = "Theoretical Means Regret")
    plt.plot(trials, trial_avg_regret, label = "Average Means Regret Across Trials")
    plt.text(NUM_TRIALS, get_avg(regret), get_avg(regret))
    worst_rew = get_regret(min(bandsim.arm_means) * NUM_SAMPLES, NUM_SAMPLES, bandsim)
    plt.plot(trials, [worst_rew for i in range(NUM_TRIALS)], label = "Pulling Worst Arm Each Time" )
    plt.xlabel("Trial")
    plt.ylabel("Regret")
    if fcn == "UCB":
        title_str = 'Regret per trial for ' + fcn + ' with ' + str(NUM_SAMPLES) + ' Samples, ' + str(NUM_ARMS) + ' Arms'
    else:
        title_str = 'Regret per trial for ' + fcn + ' with ' + str(NUM_SAMPLES) + ' Samples, ' + str(NUM_ARMS) + ' Arms, and Epsilon=' + str(EPSILON)
    plt.title(title_str)
    plt.legend()
    #plt.show(block=False)
    plt.savefig('3.png')
    #document.add_picture('3.png', width=Inches(1.25))
    #document.save('Report.docx')


def comp_ucb_eg(ucb_regret, eg_regret, bandsim):
    trials = list(range(1, NUM_TRIALS+1))
    ucb_trial_avg_regret =[get_avg(ucb_regret) for i in range(NUM_TRIALS)]
    eg_trial_avg_regret =[get_avg(eg_regret) for i in range(NUM_TRIALS)]
    plt.figure()
    plt.plot(trials, ucb_regret, label = "Actual Means Regret for UCB", color='black', alpha=0.15)
    plt.plot(trials, ucb_trial_avg_regret, label = "Average Means Regret Across Trials for UCB", color = 'black', linestyle='dashed')
    plt.plot(trials, eg_regret, label = "Actual Means Regret for e-Greedy", color = 'blue', alpha=0.15)
    plt.plot(trials, eg_trial_avg_regret, label = "Average Means Regret Across Trials for e-Greedy", color = 'blue', linestyle='dashed')
    plt.xlabel("Trial")
    plt.ylabel("Regret")
    title_str = 'Regret Comparison of UCB and e-Greedy\n' + str(NUM_SAMPLES) + ' Samples, ' + str(NUM_ARMS) + ' Arms, and Epsilon=' + str(EPSILON)
    plt.title(title_str)
    plt.legend()
    #plt.show(block=False)
    plt.savefig('4.png')
    #document=Document('Report.docx')
    #document.add_picture('4.png', width=Inches(1.25))
    #document.save('Report.docx')

def list_avg(myList):
    return np.mean(myList, axis=0)

def ucb(samples, num_arms, bandsim):
    V = 0
    r_list = []
    #Try all actions before the loop:
    for arm in range(num_arms):
        r = bandsim.pull_arm(arm)
        V += r
        r_list.append(r)
        #Arm.instances[arm].emp_mean = (r/2)
        Arm.instances[arm].emp_mean = r
        Arm.instances[arm].pulls += 1
    #Repeat Loop - must sample less times since we already pulled once on each arm
    for i in range(1, samples-NUM_ARMS):
        #Argmax
        curr_max = 0
        max_idx = None
        for arm in range(num_arms):
            arg_val = Arm.instances[arm].emp_mean + np.sqrt((2 * np.log10(i))/Arm.instances[arm].pulls)
            if arg_val > curr_max:
                curr_max = arg_val
                max_idx = arm
        
        #Pull arm with highest Hoeffding bound
        r = bandsim.pull_arm(max_idx)
        r_list.append(r)
        V += r
        na = Arm.instances[max_idx].pulls
        Arm.instances[max_idx].emp_mean = ((na * Arm.instances[max_idx].emp_mean + r)/(na +1))
        Arm.instances[max_idx].pulls += 1
    
    return V, r_list

def egreedy(samples, num_arms, e, bandsim):
    V = 0
    r_list = []
    for i in range(1, samples):
        rand_num = random.random()
        #Randomly choose an arm if random number meets criteria 
        if rand_num > (1-e) or V==0:
            max_idx = random.choice(range(len(Arm.instances)))
        #Else pick arm with highest average so far:
        else:
            max_idx = max(range(len(Arm.instances)), key=lambda i: Arm.instances[i].avg)
        
        #Get reward
        payout = bandsim.pull_arm(max_idx)
        r_list.append(payout)
        #Update average
        Arm.instances[max_idx].update_avg(payout)
        Arm.instances[max_idx].pulls += 1
        V += payout
    return V, r_list

def get_regret(V, NUM_SAMPLES, bandsim):
    '''
    #Pull the max means arm NUM_SAMPLES times
    best_val = 0
    optimal = np.argmax(bandsim.arm_means)
    for i in range(NUM_SAMPLES):
        r = bandsim.pull_arm(optimal)
        best_val += r
    return best_val - V
    '''
    
    #Assumes we get the mean each time
    return max(bandsim.arm_means)*NUM_SAMPLES - V

    #Assumes we get the absolute maximum each time: Very unlikely
    #return (max(bandsim.arm_means) + STD_DEV*np.sqrt(NUM_SAMPLES -1)) * NUM_SAMPLES - V

def get_avg(myList):
    return (sum(myList)/len(myList))

if __name__ == '__main__':
    main()

