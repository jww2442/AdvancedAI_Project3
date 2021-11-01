
import cs7313_mdp.gridworld as gw

import numpy as np
import random

from value_iteration_gridworld import Q_value


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
        expected_util_given_s_and_a += next_prob*(reward + discount * U.get(next_state.pos)) 
    
    return expected_util_given_s_and_a




if(__name__ == '__main__'): 

    mdp = gw.DiscreteGridWorldMDP(4, 4)

    mdp.add_obstacle('pit', (2,2))
    mdp.add_obstacle('pit', (1,2))
    mdp.add_obstacle('pit', (1,1))
    mdp.add_obstacle('pit', (1,0))

    mdp.add_obstacle('goal', (3,3))

    mdp.display()

    pi = policy_iteration(mdp, 0.7)

    print('\n\nU:', pi)


