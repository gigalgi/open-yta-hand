import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os

# Set the directory path and file names
data_directory = 'dataset'
train_file_name = 'train_data.csv'
test_file_name = 'test_data.csv'

# Join the directory and file names to create full file paths
train_file_path = os.path.join(data_directory, train_file_name)
test_file_path = os.path.join(data_directory, test_file_name)

# Load data from text file
def load_data(file_path):
    # Read the file using pandas
    data = pd.read_csv(file_path, sep=",", header=None)
    return data

# Load the training and testing data
df_train = load_data(train_file_path)
df_test = load_data(test_file_path)

# Separate inputs and outputs for training and testing
inputs_train = torch.tensor(df_train.iloc[:, :2].values, dtype=torch.float32)
outputs_train = torch.tensor(df_train.iloc[:, 2:].values, dtype=torch.float32)

inputs_test = torch.tensor(df_test.iloc[:, :2].values, dtype=torch.float32)
outputs_test = torch.tensor(df_test.iloc[:, 2:].values, dtype=torch.float32)

# Create DataLoader
train_dataset = TensorDataset(inputs_train, outputs_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = TensorDataset(inputs_test, outputs_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class LSTMInverseKinematics(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMInverseKinematics, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Initializing hidden state and cell state
        h0 = torch.zeros(1, x.size(0), hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), hidden_size).to(x.device)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

# Create a directory for tensorboard logs
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Initialize TensorBoard writer
writer = SummaryWriter(log_dir)

# Hyperparameters
input_size = 2
hidden_size = 64
output_size = 1
num_epochs = 100
learning_rate = 0.01

# Model, loss function, optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMInverseKinematics(input_size, hidden_size, output_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)

# Learning rate scheduler (Optional)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# Training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        outputs = model(inputs.unsqueeze(1))  # LSTM expects a 3D input
        loss = criterion(outputs, targets)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * inputs.size(0)
    
    train_loss /= len(train_loader.dataset)
    
    # Validation step
    model.eval()
    val_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs.unsqueeze(1))  # LSTM expects a 3D input
            loss = criterion(outputs, targets)
            
            val_loss += loss.item() * inputs.size(0)
            
            # Collect predictions and targets for metrics calculation
            all_predictions.append(outputs.cpu())
            all_targets.append(targets.cpu())
    
    val_loss /= len(test_loader.dataset)
    
    # Calculate metrics
    all_predictions = torch.cat(all_predictions, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    r2 = r2_score(all_targets, all_predictions)
    mse = mean_squared_error(all_targets, all_predictions)
    mae = mean_absolute_error(all_targets, all_predictions)
    rmse = mse ** 0.5
    
    # Log to TensorBoard
    writer.add_scalar('Loss/Train', train_loss, epoch)
    writer.add_scalar('Loss/Validation', val_loss, epoch)
    writer.add_scalar('R2/Validation', r2, epoch)
    writer.add_scalar('MSE/Validation', mse, epoch)
    writer.add_scalar('RMSE/Validation', rmse, epoch)
    writer.add_scalar('MAE/Validation', mae, epoch)
    writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
    
    # Log histograms of model parameters
    for name, param in model.named_parameters():
        writer.add_histogram(name, param, epoch)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, '
          f'Validation R2: {r2:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, '
          f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
    
    # Update the learning rate
    scheduler.step()
    
# Close TensorBoard writer
writer.close()

# Save the trained model
model_save_path = "weights/lstm_inverse_kinematics.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

