from collections import defaultdict
from mdp import *
from gridworld import *
from gridworld_c import *
import numpy as np
import matplotlib.pyplot as plt
import timeit
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
        self.locCount = {}

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

    def countMake(self):
        x = self.mdp._w
        #print(x)
        y = self.mdp._h
        for i in range(x+1):
            for j in range(y+1):
                #print(j)
                self.locCount[i, j] = 0
                #print(self.locCount[i, j])

    def updateCount(self, pos):
        x, y = pos
        self.locCount[x, y] += 1

    def getCount(self, pos):
        x, y = pos
        return self.locCount[x, y]

def exploreFunc(agent, s1):
    x, y = s1.pos
    n = [0, 0, 0, 0]
    if x-1 > 0:
        n[0] = agent.locCount[x - 1, y]
    if x+1 < agent.mdp._h:
        n[1] = agent.locCount[x + 1, y]
    if y+1 < agent.mdp._w:
        n[2] = agent.locCount[x, y+1]
    if y-1 > 0:
        n[3] = agent.locCount[x, y-1]
    a = min(n)
    b = n.index(a)
    if a < 10:
        if b == 0:
            return Actions.DOWN
        elif b==1:
            return Actions.UP
        elif b==2:
            return Actions.LEFT
        elif b==3:
            return Actions.RIGHT
    else:
        return max(agent.mdp.actions_at(s1), key=lambda a1: agent.find(agent.Qtable[s1, a1], agent.Nsa[s1, a1]))


def QLearn(agent: Agent, perc):
    s, a, r = agent.prevArgs()
    s1, r1 = perc
    agent.updateCount(s1.pos)
    #print((s1.pos))
    if agent.terminal(s1):
        agent.Qtable[s1, None] = r1
    if s is not None:
        agent.Nsa[s,a] += 1
        agent.Qtable[s,a] += agent.alpha(agent.Nsa[s,a]) * (r + agent.gamma
                                                * max(agent.Qtable[s1, a1] for a1 in agent.mdp.actions_at(s1)) - agent.Qtable[s, a])
    if agent.terminal(s1):
        agent.reset()
    else:
        newA = exploreFunc(agent, s1)
        agent.update(s1, newA, r1)
        #print(agent.a)
    return agent.a

def SARSALearn(agent: Agent, perc):
    s, a, r = agent.prevArgs()
    s1, r1 = perc
    agent.updateCount(s1.pos)
    a1 = exploreFunc(agent, s1)
    if agent.terminal(s1):
        agent.Qtable[s1, None] = r1
    if s is not None:
        agent.Nsa[s, a] += 1
        agent.Qtable[s1, a1] += agent.alpha(agent.Nsa[s, a]) * (r + agent.gamma * agent.Qtable[s1, a1] - agent.Qtable[s, a])
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
        #print(x)
        rTotal = 0
        count = 0
        while True:
            r = mdp.r(x)
            rTotal += r
            count += 1
            perc = (x, r)
            a1 = QLearn(agent, perc)
            if a1 is None:
                break
            x, _ = mdp.act(x, a1)
            #print(x)
        rewards.append([i + 1, rTotal, count])
    return rewards

def SARSArun(agent: Agent, n):
    print("SARSA Run")
    rewards = []
    for i in range(n):
        mdp = agent.mdp
        x = mdp.initial_state
        rTotal = 0
        count = 0
        while True:
            r = mdp.r(x)
            rTotal += r
            count += 1
            perc = (x, r)
            a1 = SARSALearn(agent, perc)
            if a1 is None:
                break
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

def printPolicies_Grid(mdp, policies, utilities=None, isQLearning=True):
    # import matplotlib
    # import matplotlib.pyplot as plt

    data = [[np.nan] * mdp.width for _ in range(mdp.height)]
    if utilities != None:
        for key, value in utilities.items():
            key = key.coords
            data[key[1]][key[0]] = round(value, 4)
    dataArray = np.array(data)

    fig, ax = plt.subplots()
    im = ax.imshow(dataArray, origin='lower', extent=(0, dataArray.shape[1], 0, dataArray.shape[0]), cmap='cool')
    if utilities != None:
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Q Value', rotation=-90, va="bottom")

    ax.set_xticks(range(mdp.width))
    ax.set_xticks(np.arange(0.5, mdp.width, 1), minor=True)

    ax.set_yticks(range(mdp.height))
    ax.set_yticks(np.arange(0.5, mdp.height, 1), minor=True)
    # ... and label them with the respective list entries
    ax.set_xticklabels([])
    ax.set_xticklabels(range(mdp.width), minor=True)

    ax.set_yticklabels([])
    ax.set_yticklabels(range(mdp.height), minor=True)

    plt.setp(ax.get_xticklabels(minor=True), rotation=15, ha="center",
             rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(minor=True), rotation=15, ha="center",
             rotation_mode="anchor")

    tPolicies = transform(mdp, policies)
    # Loop over data dims and create arrows for policies
    for i in range(len(dataArray)):
        for j in range(len(dataArray[0])):
            obs_lab = lambda p, lab, kind: lab if mdp.obs_at(kind, p) else ''
            p = (j, i)
            l_s = 'Start\n' if p == (0, 0) else ''
            l_gl = obs_lab(p, 'Goal\n', 'goal')
            l_p = obs_lab(p, 'Pit\n', 'pit')
            text = ax.text(j + .5, i + .6, dataArray[i, j],
                           ha="center", va="center", color="w", fontsize='medium', fontweight='bold')
            roomInfo = l_s + l_p + l_gl
            text = ax.text(j + 0.5, i + 0.9, roomInfo,  # abs(i - len(dataArray) + 1)
                           ha="center", va="top", color="b", fontweight='bold', fontsize='medium')
            if tPolicies[i, j] != None:
                xOffset, yOffset = tPolicies[i, j]
                plt.arrow(j + 0.5 - xOffset, i + 0.3 - yOffset, xOffset, yOffset, head_width=0.1, head_length=0.1,
                          width=0.02)
    if isQLearning:
        ax.set_title("Q Learning Results")
    else:
        ax.set_title("SARSA Results")
    ax.grid(which='major', color='k', linewidth=2)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.3)

    fig.text(0.05, 0.2, 'Gamma: ' + str(mdp.gamma))

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
    mdp = DiscreteGridWorldMDP(4, 3, -0.2)
    mdp.add_obstacle('pit', (3, 1), -1)
    mdp.add_obstacle('goal', (3, 2), 1)
    #mdp.setWalls((1, 1))
    return mdp

def genGrid(w, h, goalPos):
    mdp = DiscreteGridWorldMDP(w, h, -0.2)
    x, y = goalPos
    mdp.add_obstacle('goal', [x, y], 1)
    return mdp

def genGridC(w, h, goalPos):
    mdp = ContinuousGridWorldMDP(w, h, -0.2)
    x, y = goalPos
    mdp.add_goal([x, y], 1, reward = 1)
    return mdp

qResults = []
def testQ(mdp,  episode, trial, trials):
    for i in range(trials):
        global agent, qResults
        print("Q Learn Trial: ", trial)
        trial += 1
        agent = Agent(mdp, gamma = 0.9, alpha = 0.5)
        agent.countMake()
        results = Qrun(agent, episode)
        if qResults:
            for i in range(len(results)):
                qResults[i] = [results[i][0], qResults[i][1] + results[i][1], qResults[i][2] + results[i][2]]
        else:
            qResults = results.copy()

sarsaResults = []
def testSARSA(mdp, episode, trial, trials):
    for i in range(trials):
        global agent, sarsaResults
        print("SARSA Trial: ", trial)
        trial += 1
        agent = Agent(mdp, gamma=0.9, alpha=0.5)
        agent.countMake()
        results = SARSArun(agent, episode)
        if sarsaResults:
            for i in range(len(results)):
                sarsaResults[i] = [results[i][0], sarsaResults[i][1] + results[i][1], sarsaResults[i][2] + results[i][2]]
        else:
            sarsaResults = results.copy()

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

trials = 500
episodes = 100
trial = 1
grid1 = grid17()
grid1.display()
grid2 = genGrid(10, 10, (10, 10))
#grid2.display()
grid3 = genGrid(10, 10, (5, 5))
#grid3.display()
start = timeit.timeit()
testQ(grid1, episodes, trial, trials)
end = timeit.timeit()
print("Q Time: ", end - start)
trial = 1
start = timeit.timeit()
testSARSA(grid1, episodes, trial, trials)
end = timeit.timeit()
print("SARSA Time: ", end - start)
graphResults(qResults, sarsaResults, trials)
plt.show()

qResults = []
sarsaResults = []

start = timeit.timeit()
testQ(grid2, episodes, trial, trials)
end = timeit.timeit()
print("Q Time: ", end - start)
trial = 1
start = timeit.timeit()
testSARSA(grid2, episodes, trial, trials)
end = timeit.timeit()
print("SARSA Time: ", end - start)
graphResults(qResults, sarsaResults, trials)
plt.show()

qResults = []
sarsaResults = []

start = timeit.timeit()
testQ(grid3, episodes, trial, trials)
end = timeit.timeit()
print("Q Time: ", end - start)
trial = 1
start = timeit.timeit()
testSARSA(grid3, episodes, trial, trials)
end = timeit.timeit()
print("SARSA Time: ", end - start)
graphResults(qResults, sarsaResults, trials)
plt.show()

qResults = []
sarsaResults = []

gridc = genGridC(10, 10, (5, 5))
start = timeit.timeit()
testQ(gridc, episodes, trial, trials)
end = timeit.timeit()
print("Q Time: ", end - start)
trial = 1
start = timeit.timeit()
testSARSA(gridc, episodes, trial, trials)
end = timeit.timeit()
print("SARSA Time: ", end - start)
graphResults(qResults, sarsaResults, trials)
plt.show()