
import cs7313_mdp.gridworld as gw

import time
import numpy as np
import random

#from value_iteration_gridworld import Q_value


def policy_iteration(mdp, discount):
    
    pi = {state.pos: random.choice([gw.Actions.UP, gw.Actions.DOWN, gw.Actions.LEFT, gw.Actions.RIGHT]) for state in mdp.states}
    U = {state.pos: 0 for state in mdp.states}

    while True:

        U = policy_evaluation(pi, U, mdp, discount)
        unchanged = True

        for s in mdp.states:
            best_action = None
            best_utility = -np.inf
            for a in mdp.actions_at(s):
                util = Q_value(mdp, s, a, U, discount)
                if(util>best_utility):
                    best_utility = util
                    best_action = a
                
            if(not pi.get(s.pos) == best_action):
                pi.update({s.pos: best_action})
                unchanged = False
        
        if(unchanged):
            break

    return pi

def policy_evaluation(pi, U, mdp, discount):
    U_next = U.copy()
    for s_pos in U:
        curr_state = gw.GridState(s_pos[0], s_pos[1])
        zipped_next_state_probs = list(mdp.p(curr_state, pi.get(s_pos)))
        ex_util_curr_state = 0
        for next_state_and_prob in zipped_next_state_probs:
            next_state = next_state_and_prob[0]
            next_prob = next_state_and_prob[1]
            reward = mdp.r(None, next_state)
            ex_util_curr_state += next_prob*(reward + discount * U.get(next_state.pos))
        U_next.update({s_pos: ex_util_curr_state})
    return U_next


def Q_value(mdp, s, a, U, discount):
    zipped_next_state_probs = list(mdp.p(s, a)) 
    expected_util_given_s_and_a = 0
    for next_state_and_prob in zipped_next_state_probs:
        next_state = next_state_and_prob[0]
        next_prob = next_state_and_prob[1]
        reward = mdp.r(None, next_state)
        expected_util_given_s_and_a += next_prob*(reward + discount * U.get(next_state.pos)) ##change to 1-discount?
    
    return expected_util_given_s_and_a



def make_grid(i):

    mdp = gw.DiscreteGridWorldMDP(i, i)

    if(i<=6):

        mdp.add_obstacle('pit', (0,3))
        mdp.add_obstacle('pit', (3,0))
        mdp.add_obstacle('goal', (i-1, i-1))
    
    elif(i <=8 ):

        mdp.add_obstacle('pit', (0,3))
        mdp.add_obstacle('pit', (1, 6))
        mdp.add_obstacle('pit', (4, 1))
        mdp.add_obstacle('pit', (2, 2))

        mdp.add_obstacle('goal', (i-1, i-1))

    elif(i<=10):
        mdp.add_obstacle('pit', (0, 1))
        mdp.add_obstacle('pit', (1, 6))
        mdp.add_obstacle('pit', (4, 5))
        mdp.add_obstacle('pit', (2, 8))
        mdp.add_obstacle('pit', (8, 2))

        mdp.add_obstacle('goal', (i-1, i-1))

    return mdp

if(__name__ == '__main__'): 
    from cs7313_mdp import gridworld as gw

    times = []
    for i in [4, 5, 6, 7, 8, 9, 10]:
        mdp = make_grid(i)

        mdp.display()
        t0 = time.perf_counter()
        pi = policy_iteration(mdp, 0.7)
        t1 = time.perf_counter()

        print('Time taken: ', t1-t0, '\npi:', pi, '\n\n')
        times.append(t1-t0)
    print(times)
    


