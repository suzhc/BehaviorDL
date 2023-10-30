import torch
from torch.utils.data import DataLoader, random_split
from data_provider.data_loader import HuaweiDataset
from models.DLinear import DLinear


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the dataset
dataset = HuaweiDataset(path="/home/user/suzhao/BehaviorDL/dataset/Huawei", label_flag='energy')

# Define the hyperparameters
timesteps = 1441
num_features = 4
input_dim = timesteps * num_features
hidden_dim = 128
output_dim = 2
lr = 0.0001
epochs = 5

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


# Initialize the model, criterion and optimizer
model = DLinear(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)\
    .to(device)  # Adjust input_dim accordingly
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Training loop
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for data, labels in train_loader:
        data, labels = data.view(-1, timesteps*num_features).float().to(device), labels.long().to(device)
        
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels.squeeze())
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

# Validation loop
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, labels in val_loader:
        data, labels = data.view(-1, timesteps*num_features).float().to(device), labels.long().to(device)
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.squeeze()).sum().item()

print(f"Validation Accuracy: {100 * correct / total}%")