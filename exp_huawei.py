import torch
from torch.utils.data import DataLoader, random_split
from data_provider.data_loader import HuaweiDataset
from models.DLinear import Model as DLinear
from models.PatchTST import Model as PatchTST
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='DLinear')
parser.add_argument('--d_model', type=int, default=64)
parser.add_argument('--n_heads', type=int, default=4)
parser.add_argument('--e_layers', type=int, default=2)
parser.add_argument('--d_ff', type=int, default=256)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--factor', type=int, default=5)
parser.add_argument('--activation', type=str, default='gelu')
parser.add_argument('--seq_len', type=int, default=1441)
parser.add_argument('--enc_in', type=int, default=4)
parser.add_argument('--output_attention', type=bool, default=False)
parser.add_argument('--num_class', type=int, default=2)
parser.add_argument('--num_features', type=int, default=4)
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--output_dim', type=int, default=2)
config = parser.parse_args()



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the dataset
dataset = HuaweiDataset(path="/home/user/suzhao/BehaviorDL/dataset/Huawei", label_flag='emotion')

# Define the hyperparameters
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
model = DLinear(config).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Training loop
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for data, labels in train_loader:
        data, labels = data.float().to(device), labels.long().to(device)
        
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
        data, labels = data.float().to(device), labels.long().to(device)
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.squeeze()).sum().item()

print(f"Validation Accuracy: {100 * correct / total}%")