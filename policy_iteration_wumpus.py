import cs7313_mdp.wumpus

import random
import numpy as np

def policy_iteration(mdp, discount):
    ss = mdp.states
    pi = {state.__repr__(): random.choice(mdp.actions_at(state)) for state in ss}
    U = {state.__repr__(): 0 for state in ss}
    
    string_to_state = {state.__repr__(): state for state in ss}

    while True:

        U = policy_evaluation(pi, U, mdp, discount, string_to_state)
        unchanged = True

        for str in U:
            s = string_to_state.get(str)
            best_action = None
            best_utility = -np.inf
            for a in mdp.actions_at(s):
                util = Q_value(mdp, s, a, U, discount)
                if(util>best_utility):
                    best_utility = util
                    best_action = a
                
            if(not pi.get(str) == best_action):
                pi.update({str: best_action})
                unchanged = False
        
        if(unchanged):
            break

    return pi, U

def policy_evaluation(pi, U, mdp, discount, string_to_state):
    U_next = U.copy()
    for str in U: #changed
        #curr_state = gw.GridState(s_pos[0], s_pos[1])
        curr_state = string_to_state.get(str)#changed
        zipped_next_state_probs = list(mdp.p(curr_state, pi.get(str))) #changed
        ex_util_curr_state = 0
        for next_state_and_prob in zipped_next_state_probs:
            next_state = next_state_and_prob[0]
            next_prob = next_state_and_prob[1]
            reward = mdp.r(None, next_state)
            ex_util_curr_state += next_prob*(reward + discount * U.get(str))#changed 
        U_next.update({str: ex_util_curr_state})
    return U_next


def Q_value(mdp, s, a, U, discount):
    zipped_next_state_probs = list(mdp.p(s, a)) 
    expected_util_given_s_and_a = 0
    for next_state_and_prob in zipped_next_state_probs:
        next_state = next_state_and_prob[0]
        next_prob = next_state_and_prob[1]
        reward = mdp.r(None, next_state)
        #print(next_prob, reward, U.get(next_state.__repr__()))
        expected_util_given_s_and_a += next_prob*(reward + discount * U.get(next_state.__repr__())) #changed
    
    return expected_util_given_s_and_a


def create_wumpus(w_num):

    mdp = None

    if(w_num == 0):
        mdp = wumpus.WumpusMDP(3, 4, -100, 10, -1, -5, 10)

        mdp.add_obstacle('wumpus', [1, 0])
        mdp.add_obstacle('wumpus', [1, 1])
        mdp.add_obstacle('wumpus', [1, 2])
        mdp.add_obstacle('goal', [2, 3])
    elif(w_num == 1):
        mdp = wumpus.WumpusMDP(8, 10, -0.1)

        ## add wumpus
        mdp.add_obstacle('wumpus', [6, 9], -100) # super hurtful wumpus
        mdp.add_obstacle('wumpus', [6, 8])
        mdp.add_obstacle('wumpus', [6, 7])
        mdp.add_obstacle('wumpus', [7, 5])

        ## add pits
        mdp.add_obstacle('pit', [2, 0])
        mdp.add_obstacle('pit', [2, 1])
        mdp.add_obstacle('pit', [2, 2], -0.5) # weaker pit
        mdp.add_obstacle('pit', [5, 0])
        mdp.add_obstacle('pit', [6, 1])

        ## add goal
        mdp.add_obstacle('goal', [7, 9])

        ## add objects
        mdp.add_object('gold', [0, 9])
        mdp.add_object('gold', [7, 0])
        mdp.add_object('immune', [6, 0])

        mdp.add_object('gold', [1, 1])
        mdp.add_object('immune', [1, 2])

    return mdp

if(__name__ == '__main__'): 
    from cs7313_mdp import wumpus

    for i in [0, 1]:
        mdp = create_wumpus(i)

        mdp.display()

        #print(mdp.states)
        pi, U = policy_iteration(mdp, 0.9)
        print(pi, '\n\n')