import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os

class CNN_QNet(nn.Module):
    def __init__(self, input_channels=3, output_size=3):
        super().__init__()
        # CNN layers for processing grid-based state
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        
        # Calculate the flattened size after convolutions
        # For an input of 32x32, after conv layers it would be 128 * 14 * 14
        self.fc_input_size = self._calculate_conv_output_size(input_channels)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.fc2 = nn.Linear(512, output_size)
        
        # Batch normalization layers for better training stability
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        # Apply CNN layers with ReLU and batch normalization
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)
        
        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def _calculate_conv_output_size(self, input_channels):
        # Calculate the size of the flattened output after the conv layers
        # Assuming 32x32 input
        x = torch.zeros(1, input_channels, 32, 32)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        return x.numel()

    def save(self, file_name='cnn_model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class HybridQNet(nn.Module):
    def __init__(self, input_channels=3, feature_size=11, output_size=3):
        super().__init__()
        # CNN layers for processing grid-based state
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        
        # Calculate the flattened size after convolutions
        # For an input of 32x32, after conv layers it would be 128 * 14 * 14
        self.fc_input_size = self._calculate_conv_output_size(input_channels)
        
        # Feature processing branch
        self.feature_layer = nn.Linear(feature_size, 32)
        
        # Combined processing
        self.combined = nn.Linear(self.fc_input_size + 32, 512)
        self.out = nn.Linear(512, output_size)
        
        # Batch normalization layers for 2D data
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Layer normalization for 1D data - works with any batch size
        self.ln = nn.LayerNorm(32)
        
        # Use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, grid_state, feature_state):
        # Process grid through CNN
        x_cnn = F.relu(self.bn1(self.conv1(grid_state)))
        x_cnn = F.max_pool2d(x_cnn, 2)
        x_cnn = F.relu(self.bn2(self.conv2(x_cnn)))
        x_cnn = F.max_pool2d(x_cnn, 2)
        x_cnn = F.relu(self.bn3(self.conv3(x_cnn)))
        
        # Flatten CNN output
        x_cnn = x_cnn.view(x_cnn.size(0), -1)
        
        # Process feature vector with layer norm instead of batch norm
        x_feat = F.relu(self.ln(self.feature_layer(feature_state)))
        
        # Combine CNN and feature vector
        combined = torch.cat((x_cnn, x_feat), dim=1)
        
        # Final processing
        x = F.relu(self.combined(combined))
        x = self.out(x)
        
        return x
    
    def _calculate_conv_output_size(self, input_channels):
        # Calculate the size of the flattened output after the conv layers
        # Assuming 32x32 input
        x = torch.zeros(1, input_channels, 32, 32)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        return x.numel()

    def save(self, file_name='hybrid_model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class DuelingQNet(nn.Module):
    def __init__(self, input_channels=3, feature_size=11, output_size=3):
        super().__init__()
        # Convolutional feature extractor
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        
        # Batch normalization layers for 2D data
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Layer normalization for 1D data
        self.ln = nn.LayerNorm(32)
        
        self.fc_input_size = self._calculate_conv_output_size(input_channels)

        # Feature branch
        self.feature_layer = nn.Linear(feature_size, 32)

        # Dueling: separate fully connected streams for value and advantage
        self.value_stream = nn.Sequential(
            nn.Linear(self.fc_input_size + 32, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(self.fc_input_size + 32, 256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, grid_state, feature_state):
        x_cnn = F.relu(self.bn1(self.conv1(grid_state)))
        x_cnn = F.max_pool2d(x_cnn, 2)
        x_cnn = F.relu(self.bn2(self.conv2(x_cnn)))
        x_cnn = F.max_pool2d(x_cnn, 2)
        x_cnn = F.relu(self.bn3(self.conv3(x_cnn)))
        x_cnn = x_cnn.view(x_cnn.size(0), -1)

        x_feat = F.relu(self.ln(self.feature_layer(feature_state)))
        combined = torch.cat((x_cnn, x_feat), dim=1)

        value = self.value_stream(combined)
        advantage = self.advantage_stream(combined)
        return value + advantage - advantage.mean(dim=1, keepdim=True)

    def _calculate_conv_output_size(self, input_channels):
        # Calculate the size of the flattened output after the conv layers
        # Assuming 32x32 input
        x = torch.zeros(1, input_channels, 32, 32)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        return x.numel()

    def save(self, file_name='dueling_model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.device = self.model.device

    def train_step(self, state, action, reward, next_state, done):
        # Ensure each input is a list, so indexing works
        if not isinstance(state, list):
            state = [state]
        if not isinstance(action, list):
            action = [action]
        if not isinstance(reward, list):
            reward = [reward]
        if not isinstance(next_state, list):
            next_state = [next_state]
        if not isinstance(done, list):
            done = [done]

        # Extract grid and features
        grid_state = []
        feature_state = []
        grid_next_state = []
        feature_next_state = []
        
        for s in state:
            grid = s[0]
            # Only convert to numpy if it's not already
            if not isinstance(grid, np.ndarray):
                try:
                    grid = np.array(grid, dtype=np.float32)
                except:
                    continue  # skip invalid
            if grid.ndim != 3:
                continue  # skip invalid
            grid_state.append(grid)
            feature_state.append(s[1])

        for ns in next_state:
            grid = ns[0]
            if not isinstance(grid, np.ndarray):
                try:
                    grid = np.array(grid, dtype=np.float32)
                except:
                    continue
            if grid.ndim != 3:
                continue
            grid_next_state.append(grid)
            feature_next_state.append(ns[1])

        # Collect consistent grids
        if grid_state:
            expected_shape = grid_state[0].shape
            consistent_grid = []
            consistent_feat = []
            consistent_action = []
            consistent_reward = []
            consistent_next_grid = []
            consistent_next_feat = []
            consistent_done = []

            for i in range(len(grid_state)):
                if grid_state[i].shape == expected_shape:
                    consistent_grid.append(grid_state[i])
                    consistent_feat.append(feature_state[i])
                    consistent_action.append(action[i])
                    consistent_reward.append(reward[i])
                    consistent_next_grid.append(grid_next_state[i])
                    consistent_next_feat.append(feature_next_state[i])
                    consistent_done.append(done[i])

            grid_state = consistent_grid
            feature_state = consistent_feat
            action = consistent_action
            reward = consistent_reward
            grid_next_state = consistent_next_grid
            feature_next_state = consistent_next_feat
            done = consistent_done

        if len(grid_state) == 0:
            return  # nothing to train

        grid_state = torch.tensor(np.stack(grid_state), dtype=torch.float, device=self.device)
        feature_state = torch.tensor(np.stack(feature_state), dtype=torch.float, device=self.device)
        grid_next_state = torch.tensor(np.stack(grid_next_state), dtype=torch.float, device=self.device)
        feature_next_state = torch.tensor(np.stack(feature_next_state), dtype=torch.float, device=self.device)
        action = torch.tensor(action, dtype=torch.long, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float, device=self.device)

        pred = self.model(grid_state, feature_state)
        target = pred.clone()

        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(
                    grid_next_state[idx].unsqueeze(0),
                    feature_next_state[idx].unsqueeze(0)
                ))

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
