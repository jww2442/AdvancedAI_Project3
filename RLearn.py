from collections import defaultdict
from mdp import *
from gridworld import *
import numpy as np
import matplotlib.pyplot as plt
#For Q-learning and SARSA, use αs,a=1/ns,a, i.e., the learing rate for a state-action combination is the inverse of the number of times that action has been taken from that state.

class Agent:
    def __init__(self, mdp, gamma = 0.9, alpha = None, Rplus = 0.1, Ne = 5):
        if alpha:
            if isinstance(alpha, float):
                self.alpha = lambda n: alpha
            else:
                self.alpha = alpha
        else:
            self.alpha = lambda n: 1 / (1 + n)
        self.mdp = mdp
        self.gamma = gamma
        self.mdp.gamma = gamma
        self.Rplus = Rplus
        self.Ne = Ne
        self.s = None
        self.a = None
        self.r = None
        self.Qtable = defaultdict(float)
        self.Nsa = defaultdict(int)

    def prevArgs(self):
        return self.s, self.a, self.r

    def update(self, s, a, r):
        self.a = a
        self.s = s
        self.r = r

    def reset(self):
        self.s = None
        self.a = None
        self.r = None

    def terminal(self, state):
        if self.mdp.is_terminal(state):
            return True
        else:
            return False

    def find(self, u, n):
        if n < self.Ne:
            return self.Rplus
        else:
            return u


def QLearn(agent: Agent, perc):
    s, a, r = agent.prevArgs()
    agent.Qtable = agent.Qtable
    agent.Nsa = agent.Nsa
    s1, r1 = perc
    if agent.terminal(s1):
        agent.Qtable[s1, None] = r1
    if s is not None:
        agent.Nsa[s,a] += 1
        agent.Qtable[s,a] += agent.alpha(agent.Nsa[s,a]) * (r + agent.gamma
                                                * max(agent.Qtable[s1, a1] for a1 in agent.mdp.actions_at(s1)) - agent.Qtable[s, a])
    if agent.terminal(s1):
        agent.reset()
    else:
        newA = max(agent.mdp.actions_at(s1), key=lambda a1: agent.find(agent.Qtable[s1, a1], agent.Nsa[s1, a1]))
        agent.update(s1, newA, r1)
    return agent.a

def SARSALearn(agent: Agent, perc):
    s, a, r = agent.prevArgs()
    qtable = agent.Qtable
    nsa = agent.Nsa
    s1, r1 = perc
    a1 = max(agent.mdp.actions_at(s1), key = lambda a1: agent.find(qtable[s1, a1], nsa[s1, a1]))
    if agent.terminal(s1):
        qtable[s1, None] = r1
    if s is not None:
        nsa[s, a] += 1
        qtable[s1, a1] += agent.alpha(nsa[s, a]) * (r + agent.gamma * qtable[s1, a1] - qtable[s, a])
    if agent.terminal(s1):
        agent.reset()
    else:
        agent.update(s1, a1, r1)
    return agent.a

def Qrun(agent: Agent, n):
    print("Q Run")
    rewards = []
    for i in range(n):
        mdp = agent.mdp
        x = mdp.initial_state
        print(x)
        rTotal = 0
        count = 0
        r = 0
        while True:
            r = mdp.r(x, x)
            rTotal += r
            count += 1
            perc = (x, r)
            a1 = QLearn(agent, perc)
            if a1 is None:
                break
            x, _ = mdp.act(x, a1)
        rewards.append([i + 1, rTotal, count])
    return rewards

def SARSArun(agent: Agent, n):
    rewards = []
    for i in range(n):
        mdp = agent.mdp
        x = mdp.initial_state
        rTotal = 0
        count = 0
        while a1 is not None:
            r = mdp.r(x, a1)
            rTotal += r
            count += 1
            perc = (x, r)
            a1 = SARSALearn(agent, perc)
            x, _ = mdp.act(x, a1)
        rewards.append([i + 1, rTotal, count])
    return rewards

def transform(mdp, policy):
    tpolicy = [[None] * mdp.width for i in range(mdp.height)]
    for key, val in policy.items():
        key = key.coords
        if val == None:
            continue
        val = val.value
        if val == Actions.UP.value:
            tpolicy[key[1]][key[0]] = (0, 0.1)
        elif val == Actions.DOWN.value:
            tpolicy[key[1]][key[0]] = (0, -.1)
        elif val == Actions.LEFT.value:
            tpolicy[key[1]][key[0]] = (-.1, 0)
        elif val == Actions.RIGHT.value:
            tpolicy[key[1]][key[0]] = (.1, 0)
    return np.array(tpolicy)

def result(agent):
    steps = []
    for key, val in dict(agent.Qtable).items():
        if key[0] in steps:
            if val > steps[key[0]][1]:
                steps[key[0]] = (key[1], val)
        else:
            steps[key[0]] = (key[1], val)
    policy = {}
    qVal = {}
    for key, val in steps.items():
        policy[key] = val[0]
        qVal[key] = val[1]
    return policy, qVal

def grid17():
    mdp = DiscreteGridWorldMDP(4, 3, -0.04)
    mdp.add_obstacle('pit', [3, 1], -1)
    mdp.add_obstacle('goal', [3, 2], 1)
    #mdp.setWalls((1, 1))
    return mdp

def genGrid(w, h, goalPos):
    mdp = DiscreteGridWorldMDP(w, h)
    x, y = goalPos
    mdp.add_obstacle('goal', [x, y], 1)
    return mdp

def testQ(mdp,  episode, trial):
    global agent, qResults
    print("Q Learn Trial: ", trial)
    trial += 1
    agent = Agent(mdp, gamma = 0.9, alpha = 0.5)
    results = Qrun(agent, episode)
    if qResults:
        for i in range(len(results)):
            qResults[i] = [results[i][0], qResults[i][1] + results[i][1], qResults[i][2] + results[i][2]]
    else:
        qResults = results.copy()

def testSARSA(mdp, episode, trial):
    global agent, sarsaResults
    print("SARSA Trial: ", trial)
    trial += 1
    agent = Agent(mdp, gamma=0.9, alpha=0.5)
    results = SARSArun(agent, episode)
    if sarsaResults:
        for i in range(len(results)):
            sarsaResults[i] = [results[i][0], sarsaResults[i][1] + results[i][1], sarsaResults[i][2] + results[i][2]]
    else:
        sarsaResults = results.copy()

trials = 1
episodes = 1
trial = 1
grid1 = grid17()
#grid1.display()
grid2 = genGrid(10, 10, (9, 9))
#grid2.display()
grid3 = genGrid(10, 10, (4, 4))
#grid3.display()
print(testQ(grid1, episodes, trial), number = trials)

def graphResults(qResults, sarsaResults, trials):
    for i in range(len(qResults)):
        qResults[i][1] /= trials
        sarsaResults[i][1] /= trials
        qResults[i][2] /= trials
        sarsaResults[i][2] /= trials
    episodeListQ, rewardListQ, stepListQ = zip(*qResults)
    episodeListSARSA, rewardListSARSA, stepListSARSA = zip(*sarsaResults)

    plt.figure('4x3 Reward vs Episodes')
    qPlot, = plt.plot(episodeListQ, rewardListQ, label='Q Learning')
    sarsaPlot, = plt.plot(episodeListSARSA, rewardListSARSA, label='SARSA')
    plt.xlabel('Episode')
    plt.ylabel('Avg Reward')
    # plt.ylabel('Avg Steps Per Episode')
    plt.title('Q Learning - Avg Reward vs Episode')
    # plt.title('Q Learning - Avg Steps Per Episode vs Episode')
    plt.legend([qPlot, sarsaPlot], ['Q Learning', 'SARSA'], loc='upper right')

    plt.figure('4x3 Steps vs Episodes')
    qPlot, = plt.plot(episodeListQ, stepListQ, label='Q Learning')
    sarsaPlot, = plt.plot(episodeListSARSA, stepListSARSA, label='SARSA')
    plt.xlabel('Episode')
    # plt.ylabel('Avg Reward')
    plt.ylabel('Avg Steps Per Episode')
    # plt.title('Q Learning - Avg Reward vs Episode')
    plt.title('Q Learning - Avg Steps Per Episode vs Episode')
    plt.legend([qPlot, sarsaPlot], ['Q Learning', 'SARSA'], loc='upper right')


def graphResults_Expanded(qResults, qResults2, sarsaResults, sarsaResults2, trials):
    for i in range(len(qResults)):
        qResults[i][1] /= trials
        sarsaResults[i][1] /= trials
        qResults[i][2] /= trials
        sarsaResults[i][2] /= trials
    episodeListQ, rewardListQ, stepListQ = zip(*qResults)
    episodeListSARSA, rewardListSARSA, stepListSARSA = zip(*sarsaResults)
    episodeListQ2, rewardListQ2, stepListQ2 = zip(*qResults2)
    episodeListSARSA2, rewardListSARSA2, stepListSARSA2 = zip(*sarsaResults2)

    plt.figure('4x3 Reward vs Episodes')
    qPlot, = plt.plot(episodeListQ, rewardListQ, label='Q Learning')
    sarsaPlot, = plt.plot(episodeListSARSA, rewardListSARSA, label='SARSA')

    qPlot2, = plt.plot(episodeListQ2, rewardListQ2, label='Q Learning (No Noise)')
    sarsaPlot2, = plt.plot(episodeListSARSA2, rewardListSARSA2, label='SARSA (No Noise)')
    plt.xlabel('Episode')
    plt.ylabel('Avg Reward')
    # plt.ylabel('Avg Steps Per Episode')
    plt.title('Q Learning - Avg Reward vs Episode')
    # plt.title('Q Learning - Avg Steps Per Episode vs Episode')
    plt.legend([qPlot, sarsaPlot, qPlot2, sarsaPlot2],
               ['Q Learning', 'SARSA', 'Q Learning (No Noise)', 'SARSA (No Noise)'], loc='upper right')

    plt.figure('4x3 Steps vs Episodes')
    qPlot, = plt.plot(episodeListQ, stepListQ, label='Q Learning')
    sarsaPlot, = plt.plot(episodeListSARSA, stepListSARSA, label='SARSA')
    qPlot2, = plt.plot(episodeListQ2, stepListQ2, label='Q Learning 2 ')
    sarsaPlot2, = plt.plot(episodeListSARSA2, stepListSARSA2, label='SARSA 2')
    plt.xlabel('Episode')
    # plt.ylabel('Avg Reward')
    plt.ylabel('Avg Steps Per Episode')
    # plt.title('Q Learning - Avg Reward vs Episode')
    plt.title('Q Learning - Avg Steps Per Episode vs Episode')
    plt.legend([qPlot, sarsaPlot, qPlot2, sarsaPlot2],
               ['Q Learning', 'SARSA', 'Q Learning (No Noise)', 'SARSA (No Noise)'], loc='upper right')
    # plt.legend([qPlot, sarsaPlot], ['Q Learning', 'SARSA'], loc='upper right')
