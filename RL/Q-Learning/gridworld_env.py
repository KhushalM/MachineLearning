import gymnasium as gym
import numpy as np
import pygame
from gymnasium.spaces import Discrete
from collections import defaultdict
import time

CELL_SIZE = 100
MARGIN = 10



def get_coords(row, col, loc='center'):
    xc = (col + 1.5) * CELL_SIZE
    yc = (row + 1.5) * CELL_SIZE
    if loc == 'center':
        return xc, yc
    elif loc == 'interior_corners':
        half_size = CELL_SIZE // 2 - MARGIN
        xl, xr = xc - half_size, xc + half_size
        yt, yb = yc - half_size, yc + half_size
        return [(xl, yt), (xr, yt), (xr, yb), (xl, yb)]
    elif loc == 'interior_triangle':
        x1, y1 = xc, yc + CELL_SIZE // 3
        x2, y2 = xc + CELL_SIZE // 3, yc - CELL_SIZE // 3
        x3, y3 = xc - CELL_SIZE // 3, yc - CELL_SIZE // 3
        return [(x1, y1), (x2, y2), (x3, y3)]


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, num_rows=4, num_cols=6, delay=0.5, render_mode=None):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.delay = delay

        self.action_space = Discrete(4)  # Actions: up(0), right(1), down(2), left(3)
        self.observation_space = Discrete(num_rows * num_cols)

        move_up = lambda row, col: (max(row - 1, 0), col)
        move_down = lambda row, col: (min(row + 1, num_rows - 1), col)
        move_left = lambda row, col: (row, max(col - 1, 0))
        move_right = lambda row, col: (row, min(col + 1, num_cols - 1))

        self.action_defs = {0: move_up, 1: move_right,
                            2: move_down, 3: move_left}

        # Map grid positions to states and vice versa
        self.grid2state_dict = {(s // num_cols, s % num_cols): s for s in range(num_rows * num_cols)}
        self.state2grid_dict = {s: (s // num_cols, s % num_cols) for s in range(num_rows * num_cols)}

        # Define terminal states (gold and traps)
        gold_cell = (num_rows // 2, num_cols - 2)
        trap_cells = [
            ((gold_cell[0] + 1), gold_cell[1]),
            (gold_cell[0], gold_cell[1] - 1),
            ((gold_cell[0] - 1), gold_cell[1])
        ]
        self.gold_state = self.grid2state_dict[gold_cell]
        self.trap_states = [self.grid2state_dict[cell] for cell in trap_cells]
        self.terminal_states = [self.gold_state] + self.trap_states

        # Rendering configuration
        self.render_mode = render_mode
        self.window_surface = None
        self.clock = None

    def reset(self):
        """Reset the environment."""
        self.s = 0
        return self.s

    def step(self, action):
        """Take a step in the environment."""
        row, col = self.state2grid_dict[self.s]
        next_row, next_col = self.action_defs[action](row, col)
        next_s = self.grid2state_dict[(next_row, next_col)]

        if next_s in self.terminal_states:
            reward = 1.0 if next_s == self.gold_state else -1.0
            done = True
            return next_s, reward, done
        
        reward = 0.0
        done = False
        self.s = next_s

        return next_s, reward, done

    def render(self):
        """Render the environment."""
        if self.render_mode is None:
            raise ValueError("Render mode not specified during initialization.")

        screen_width = (self.num_cols + 2) * CELL_SIZE
        screen_height = (self.num_rows + 2) * CELL_SIZE

        if self.window_surface is None:
            try:
                pygame.init()
                print("Pygame initialized successfully")
                print(f"Pygame display drivers: {pygame.display.get_driver()}")
                pygame.display.set_caption("GridWorld")
                self.window_surface = pygame.display.set_mode((screen_width, screen_height))
                print("Window surface created successfully")
                self.clock = pygame.time.Clock()
            except Exception as e:
                print(f"Error initializing Pygame: {e}")
                raise

        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        # Draw background and grid lines on a new surface
        surface = pygame.Surface((screen_width, screen_height))
        
        # Fill background with white color
        surface.fill((255, 255, 255))

        # Draw grid lines and border
        for row in range(self.num_rows + 1):
            pygame.draw.line(surface,
                           color=(0, 0, 0),
                           start_pos=(CELL_SIZE, CELL_SIZE * (row + 1)),
                           end_pos=(CELL_SIZE * (self.num_cols + 1), CELL_SIZE * (row + 1)))

        for col in range(self.num_cols + 1):
            pygame.draw.line(surface,
                           color=(0, 0, 0),
                           start_pos=(CELL_SIZE * (col + 1), CELL_SIZE),
                           end_pos=(CELL_SIZE * (col + 1), CELL_SIZE * (self.num_rows + 1)))

        # Draw the starting position (state 0)
        start_row, start_col = self.state2grid_dict[0]  # State 0 is the starting position
        start_rect = pygame.Rect(
            (start_col + 1) * CELL_SIZE + 5,
            (start_row + 1) * CELL_SIZE + 5,
            CELL_SIZE - 10,
            CELL_SIZE - 10
        )
        pygame.draw.rect(surface, (0, 200, 0), start_rect)  # Green color

        # Draw the gold (goal) cell
        gold_row, gold_col = self.state2grid_dict[self.gold_state]
        gold_rect = pygame.Rect(
            (gold_col + 1) * CELL_SIZE + 5,
            (gold_row + 1) * CELL_SIZE + 5,
            CELL_SIZE - 10,
            CELL_SIZE - 10
        )
        pygame.draw.rect(surface, (255, 215, 0), gold_rect)  # Gold color

        # Draw the trap cells
        for trap_state in self.trap_states:
            trap_row, trap_col = self.state2grid_dict[trap_state]
            trap_rect = pygame.Rect(
                (trap_col + 1) * CELL_SIZE + 5,
                (trap_row + 1) * CELL_SIZE + 5,
                CELL_SIZE - 10,
                CELL_SIZE - 10
            )
            pygame.draw.rect(surface, (255, 0, 0), trap_rect)  # Red color

        # Draw the agent
        agent_row, agent_col = self.state2grid_dict[self.s]
        agent_center = (
            (agent_col + 1.5) * CELL_SIZE,
            (agent_row + 1.5) * CELL_SIZE
        )
        pygame.draw.circle(surface, (0, 0, 255), agent_center, CELL_SIZE // 3)  # Blue circle

        # Update the display
        self.window_surface.blit(surface, (0, 0))
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])
        pygame.time.wait(100)  # Add a small delay to make the window visible

    def close(self):

        if self.window_surface is not None:
            import pygame
            pygame.quit()  # Quit pygame to release resources
            self.window_surface = None
            self.clock = None



if __name__ == "__main__":
    # Initialize the GridWorld environment with human rendering mode
    env = GridWorldEnv(num_rows=5, num_cols=6, delay=0.1, render_mode="human")
    
    # Set maximum steps and episode time limit
    max_steps = 20  # Maximum number of steps before closing
    max_episodes = 3  # Maximum number of episodes to run
    
    try:
        # Run episodes
        for episode in range(max_episodes):
            state = env.reset()  # Reset the environment
            print(f"Episode {episode+1}, Starting state: {state}")
            
            step_count = 0
            while step_count < max_steps:
                # Take a random action
                action = np.random.choice(env.action_space.n)
                next_state, reward, done = env.step(action)
                
                print(f"Step {step_count+1}: Action: {action}, Next State: {next_state}, Reward: {reward}, Done: {done}")
                
                # Render the environment (human mode)
                env.render()
                
                # Add delay between steps for slower execution
                time.sleep(0.2)
                
                step_count += 1
                
                if done:
                    print(f"Episode {episode+1} finished after {step_count} steps!")
                    time.sleep(1)  # Pause between episodes
                    break
            
            # If we've reached the maximum number of episodes, wait a bit longer before closing
            if episode == max_episodes - 1:
                print("Maximum number of episodes reached. Closing in 3 seconds...")
                time.sleep(3)
    finally:
        # Close the environment after use
        env.close()
        pygame.quit()
        print("Environment closed.")
