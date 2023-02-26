import numpy as np

from tqdm import tqdm

class PolicyFinder:
    """
    This class is used to find the best policy for a given dataset using the sarsa
    lambda algorithm.
    """
    
    def __init__(self, data: np.ndarray, nb_state: int, nb_action: int, discount: float,
                 learning: float, decay: float, epsilon: float) -> None:
        """
        This function initiates the object using various parameters entered by the user.
        """
        self.data      = data
        self.nb_state  = nb_state
        self.nb_action = nb_action

        self.discount   = discount
        self.learning   = learning
        self.decay      = decay
        self.epsilon    = epsilon

        self.Q = np.zeros((nb_state, nb_action))


    def epsilon_greedy(self, curr_state: int) -> int:
        """
        This function implements the epsilon greedy policy to choose the next action.
        It takes the current state as argument.
        """
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.randint(0, self.nb_action - 1)
        else:
            action = np.argmax(self.Q[curr_state])
        return action


    def sarsa_lambda(self) -> None:
        """
        This function implements the sarsa lambda algorithm to compute Q.
        """
        N = np.zeros((self.nb_state, self.nb_action))

        for row in tqdm(self.data, total = self.data.shape[0], desc="Data"):
            s, a, r, sp = row
            s, a, sp = s-1, a-1, sp-1
            ap = self.epsilon_greedy(sp)

            N[s][a] += 1

            delta    = r + self.discount * self.Q[sp][ap] - self.Q[s][a]
            self.Q  += self.learning * delta * N
            N       *= self.discount * self.decay
    

    def save_policy_text(self, dir: str) -> None:
        """
        This function saves the policy as a .policy file.
        """
        with open(dir + ".policy", "w") as f:
            for state in range(self.nb_state):
                action = np.argmax(self.Q[state]) + 1
                f.write(str(action) + "\n")
