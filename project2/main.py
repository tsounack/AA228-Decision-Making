import numpy as np
import time

from policyfinder import PolicyFinder

# SMALL

data_small = np.genfromtxt('data/small.csv', delimiter=',', dtype=int)[1:]
small = PolicyFinder(data = data_small, 
                     nb_state = 100, 
                     nb_action = 4, 
                     discount = 0.95,
                     learning = 0.1,
                     decay = 0.5,
                     epsilon = 0.1
                     )

t1_small = time.time()
small.sarsa_lambda()
t2_small = time.time()
small.save_policy_text("answer/small")
print("Time for Small: " + format(t2_small-t1_small, ".2f") + " seconds")



# MEDIUM

data_medium = np.genfromtxt('data/medium.csv', delimiter=',', dtype=int)[1:]
medium = PolicyFinder(data = data_medium, 
                     nb_state = 50000, 
                     nb_action = 7, 
                     discount = 0.95,
                     learning = 0.1,
                     decay = 0.5,
                     epsilon = 0.1
                     )

t1_medium = time.time()
medium.sarsa_lambda()
t2_medium = time.time()
medium.save_policy_text("answer/medium")
print("Time for Medium: " + str(t2_medium-t1_medium))



# LARGE

data_large = np.genfromtxt('data/large.csv', delimiter=',', dtype=int)[1:]
large = PolicyFinder(data = data_large, 
                     nb_state = 312020, 
                     nb_action = 9, 
                     discount = 0.95,
                     learning = 0.1,
                     decay = 0.5,
                     epsilon = 0.1
                     )

t1_large = time.time()
large.sarsa_lambda()
t2_large = time.time()
large.save_policy_text("answer/large")
print("Time for Large: " + str(t2_large-t1_large))