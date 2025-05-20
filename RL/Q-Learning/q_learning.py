from agent import Agent
from gridworld_env import GridWorldEnv
from collections import namedtuple
import numpy as np

np.random.seed(42)
Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))

def run_q_learning(agent,env, num_episodes=100):
    history = []
    for episode in range(num_episodes):
        state = env.reset()
        env.render()
        final_reward, n_moves = 0.0,0

        while True:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent._learn(Transition(state, action, reward, next_state, done))
            env.render()
            state = next_state
            
            n_moves += 1
            final_reward += reward  # Accumulate reward throughout the episode
            if done:
                break
        history.append((final_reward, n_moves))
        print(f"Episode {episode+1}: reward={final_reward}, moves={n_moves}")
    return history
    
if __name__ == "__main__":
    env = GridWorldEnv(num_rows=5, num_cols=6, delay=0.5, render_mode="human")
    agent = Agent(env)
    run_q_learning(agent,env)
    env.close()