
import numpy as np

from cs7313_mdp import wumpus

def value_iter(mdp, err, discount):

    U_prime = {state.__repr__(): 0 for state in mdp.states}
    #print(U_prime)
    
    string_to_state = {state.__repr__(): state for state in mdp.states}

    while(True): 

        U = U_prime.copy()
        max_delta = 0

        for str in U: #s is a STRING!

            s = string_to_state.get(str)
            # if(mdp.is_terminal(s)):
            #     q_val_at_s = mdp.r(s, s) #first param not used
            #     print(q_val_at_s)
            if(False):
                pass
            else:
                q_val_at_s = -np.inf
                for a in mdp.actions_at(s): 
                    q_given_a = Q_value(mdp, s, a, U, discount)
                    if (q_given_a > q_val_at_s):
                        q_val_at_s = q_given_a

            U_prime.update({str: q_val_at_s})
            
            abs_diff = abs(U_prime.get(str) - U.get(str)) 
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
        expected_util_given_s_and_a += next_prob*(reward + (1-discount) * U.get(next_state.__repr__())) #changed
        #print(expected_util_given_s_and_a)
    
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

        #mdp.display()

        #print(mdp.states)
        U, U_prime = value_iter(mdp, 0.2, 0.9)
        print(U)

