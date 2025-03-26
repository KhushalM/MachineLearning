from collections import defaultdict
import numpy as np

class Agent:
    def __init__(self,env,lr=0.1,df=0.9,eps=0.9,eps_min=0.1,eps_decay=0.95):
        self.env = env
        self.lr = lr
        self.df = df
        self.eps = eps
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.q = defaultdict(lambda: np.zeros(self.env.action_space.n))

    def choose_action(self, state):
        if np.random.random() < self.eps:
            action = np.random.randint(self.env.action_space.n)
        else:
            q_values = self.q[state]
            perm = np.random.permutation(self.env.action_space.n)
            q_values = [q_values[a] for a in perm]
            perm_q_argmax = np.argmax(q_values)
            action = perm[perm_q_argmax]
        return action

    def _learn(self, transition):
        s, a, r, next_s,done = transition
        q_val = self.q[s][a]

        if done:
            q_target = r
        else:
            q_target = r + self.df * np.max(self.q[next_s])
        
        self.q[s][a] += self.lr * (q_target - q_val)
        self._adjust_epsilon()
    
    def _adjust_epsilon(self):
        if self.eps > self.eps_min:
            self.eps *= self.eps_decay