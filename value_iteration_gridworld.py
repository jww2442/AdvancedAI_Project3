import numpy as np

from cs7313_mdp import mdp

def value_iter(mdp, err, discount):

    U_prime = {state.pos: 0 for state in mdp.states}

    while(True): 

        U = U_prime.copy()
        max_delta = 0

        for s in mdp.states: 
            
            # if(mdp.is_terminal(s)):
            #     q_val_at_s = mdp.r(s, s) #first param not used
            if(False):
                pass
            else:
                q_val_at_s = -np.inf
                for a in mdp.actions_at(s): 
                    q_given_a = Q_value(mdp, s, a, U, discount)
                    if (q_given_a > q_val_at_s):
                        q_val_at_s = q_given_a

            U_prime.update({s.pos: q_val_at_s})
            
            abs_diff = abs(U_prime.get(s.pos) - U.get(s.pos)) 
            if(abs_diff > max_delta):
                max_delta = abs_diff

        if(max_delta <= err*(1-discount)/discount):
            break

    return U, U_prime
            
def Q_value(mdp, s, a, U, discount):
    zipped_next_state_probs = list(mdp.p(s, a)) 
    expected_util_given_s_and_a = 0
    for next_state_and_prob in zipped_next_state_probs:
        next_state = next_state_and_prob[0]
        next_prob = next_state_and_prob[1]
        reward = mdp.r(s, next_state)
        expected_util_given_s_and_a += next_prob*(reward + discount * U.get(next_state.pos)) 
        #print(expected_util_given_s_and_a)
    
    return expected_util_given_s_and_a




if(__name__ == '__main__'): 
    from cs7313_mdp import gridworld as gw

    mdp = gw.DiscreteGridWorldMDP(4, 4, 0, 0, -1)

    mdp.add_obstacle('pit', (0,3))
    mdp.add_obstacle('pit', (3,0))

    mdp.display()

    U, U_prime = value_iter(mdp, 0.01, 0.7)

    print('\n\nU:', U, '\n\nUprime:', U_prime, '\n\n')
