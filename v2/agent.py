import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import DuelingQNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 128
LR = 0.001  # Lower learning rate for better stability

class Agent:
    def __init__(self, params=None):
        self.n_games = 0
        
        # Default parameters
        self.params = {
            'lr': 0.001,
            'gamma': 0.95,
            'batch_size': 128,
            'epsilon_decay': 400,  # Games to decay epsilon to minimum
            'epsilon_min': 0.02,   # Minimum exploration rate
            'memory_size': 100_000
        }
        
        # Override with provided params
        if params:
            self.params.update(params)
        
        self.epsilon = 1.0  # Start with full exploration
        self.gamma = self.params['gamma']
        self.memory = deque(maxlen=self.params['memory_size'])
        
        # Use dueling network
        self.model = DuelingQNet(input_channels=3, feature_size=11, output_size=3)
        self.trainer = QTrainer(self.model, lr=self.params['lr'], gamma=self.gamma)
        
        # Grid size for state representation
        self.grid_size = 32
        self.block_size = 20  # Same as game's BLOCK_SIZE

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # The 11 boolean features from v1
        features = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
        ]
        
        # Convert boolean features to numpy array
        features = np.array(features, dtype=int)
        
        # Create a grid-based representation for CNN with 3 channels
        grid = np.zeros((3, self.grid_size, self.grid_size), dtype=np.float32)
        
        # Convert game coordinates to grid indices
        def game_to_grid(point):
            grid_x = min(max(int(point.x / game.w * self.grid_size), 0), self.grid_size - 1)
            grid_y = min(max(int(point.y / game.h * self.grid_size), 0), self.grid_size - 1)
            return grid_x, grid_y
        
        # Place snake head in channel 0
        head_x, head_y = game_to_grid(game.head)
        grid[0, head_y, head_x] = 1
        
        # Place snake body in channel 1
        for segment in game.snake[1:]:
            grid_x, grid_y = game_to_grid(segment)
            grid[1, grid_y, grid_x] = 1
            
        # Place food in channel 2
        food_x, food_y = game_to_grid(game.food)
        grid[2, food_y, food_x] = 1
        
        return grid, features

    def remember(self, state, action, reward, next_state, done):
        # Store states as tuples containing both grid and features
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > self.params['batch_size']:
            mini_sample = random.sample(self.memory, self.params['batch_size'])
        else:
            mini_sample = self.memory
            
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # Gradual epsilon decay with a low floor
        self.epsilon = max(self.params['epsilon_min'], 
                          1.0 - self.n_games/self.params['epsilon_decay'])
        final_move = [0, 0, 0]
        if random.random() < self.epsilon:
            final_move[random.randint(0, 2)] = 1
        else:
            grid, features = state
            
            # Convert to tensors
            grid_tensor = torch.tensor(grid, dtype=torch.float, device=self.model.device)
            features_tensor = torch.tensor(features, dtype=torch.float, device=self.model.device)
            
            # Add batch dimension
            if len(grid_tensor.shape) == 3:
                grid_tensor = grid_tensor.unsqueeze(0)
            if len(features_tensor.shape) == 1:
                features_tensor = features_tensor.unsqueeze(0)
            
            with torch.no_grad():
                prediction = self.model(grid_tensor, features_tensor)
            
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train(params=None, headless=False, max_games=None):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent(params)
    game = SnakeGameAI(headless=headless)
    game_count = 0
    
    while True:
        # Get current state with both grid and features
        state_old = agent.get_state(game)

        # Get move based on current state
        final_move = agent.get_action(state_old)

        # Perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # Train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # Remember for replay memory
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # Train long memory (replay) and reset game
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            # Save model if high score
            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            # Update plots
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            
            # Only render plot if not headless
            if not headless:
                plot(plot_scores, plot_mean_scores)
            
            # Exit if max_games reached
            if max_games and agent.n_games >= max_games:
                return {
                    'max_score': record,
                    'avg_score': mean_score,
                    'scores': plot_scores
                }


if __name__ == '__main__':
    # Regular training with visualization
    train()
