import pygame
import random
import math
from enum import Enum
from collections import namedtuple, deque
import numpy as np

pygame.init()
font = pygame.font.Font('arial.ttf', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 40  # Slightly slower than before

class SnakeGameAI:
    def __init__(self, w=640, h=480, headless=False):
        self.w = w
        self.h = h
        self.headless = headless
        # init display only if not headless
        if not headless:
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption('Snake AI v2')
        self.clock = pygame.time.Clock()
        
        # For loop detection
        self.position_history = deque(maxlen=30)  # Store recent positions
        self.position_count = {}  # Count visits to each position
        self.loop_threshold = 4  # Increased threshold to allow more revisits
        
        self.reset()

    def reset(self):
        # init game state
        self.direction = Direction.RIGHT
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        
        # Reset distance tracking
        self.last_distance = self._calculate_distance(self.head, self.food)
        
        # Reset metrics for detecting loops and stuck behavior
        self.moves_without_food = 0
        self.position_history.clear()
        self.position_count = {}

    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def _is_in_loop(self):
        """Check if the snake is moving in a loop pattern"""
        # Convert point to string for use as dictionary key
        pos_key = f"{self.head.x},{self.head.y}"
        
        # Count this position
        self.position_count[pos_key] = self.position_count.get(pos_key, 0) + 1
        
        # Check if we've visited this position too many times recently
        return self.position_count[pos_key] >= self.loop_threshold

    def play_step(self, action):
        self.frame_iteration += 1
        self.moves_without_food += 1
        
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. Store the previous head position for distance calculation
        prev_head = self.head
        prev_distance = self._calculate_distance(prev_head, self.food)
        
        # 3. move
        self._move(action) # update the head
        self.snake.insert(0, self.head)
        
        # Add current position to history
        self.position_history.append((self.head.x, self.head.y))
        
        # 4. check if game over
        reward = 0
        game_over = False
        
        # Game over conditions including timeout - INCREASED DEATH PENALTY
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            reward = -10
            game_over = True
            return reward, game_over, self.score
        
        # 5. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self.moves_without_food = 0
            
            # Reset position tracking when food is eaten
            self.position_count.clear()
            
            self._place_food()
        else:
            self.snake.pop()
            
            # 6. Calculate distance-based reward - REDUCED PENALTIES
            current_distance = self._calculate_distance(self.head, self.food)
            
            # Reward for getting closer to food, penalize for getting further (reduced penalty)
            # if current_distance < prev_distance:
            #     reward -= 0.1  # Small reward for getting closer
            # else:
            #     reward += 0.01  # Reduced penalty for moving away
            
            # 7. Detect and penalize loops (reduced penalty)
            # if self._is_in_loop():
            #     reward += 0.05  # Reduced penalty for repetitive movement
            
            # Additional increasing penalty for taking too long without food (reduced)
            # if self.moves_without_food > 80:  # Increased threshold
            #     reward += 0.005 * (self.moves_without_food - 80) / 20  # Reduced penalty
            
        # 8. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
            
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
            
        # hits itself
        if pt in self.snake[1:]:
            return True
            
        return False

    def _calculate_distance(self, point1, point2):
        """Calculate Manhattan distance between two points (more appropriate for grid movement)"""
        return abs(point1.x - point2.x) + abs(point1.y - point2.y)

    def _update_ui(self):
        if self.headless:
            return
            
        self.display.fill(BLACK)

        # Draw snake
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        # Draw food
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        # Draw score
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        # [straight, right, left]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)
