import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, random_split
from data_provider.data_loader import HuaweiDataset
import numpy as np
from models.DLinear import Model as DLinear
from models.PatchTST import Model as PatchTST
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='DLinear')
parser.add_argument('--label_flag', type=str, default='emotion')
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

parser.add_argument('--random_seed', type=int, default=42)
config = parser.parse_args()

torch.manual_seed(config.random_seed)

models = {
    'DLinear': DLinear,
    'PatchTST': PatchTST
}


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the dataset
dataset = HuaweiDataset(path="/home/user/suzhao/BehaviorDL/dataset/Huawei", 
                        label_flag=config.label_flag)

# Define the hyperparameters
lr = 0.0001
epochs = 10

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders
def get_sampler(dataset):
    weights = [1 if i == 0 else 9 for i in dataset]
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=weights, 
        num_samples=len(dataset),
        replacement=True
    )
    return sampler

train_loader = DataLoader(train_dataset, batch_size=64, 
                          sampler=get_sampler(train_dataset))
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


# Initialize the model, criterion and optimizer
print(f"Using {config.model} model")
print(config)
model = models[config.model](config).to(device)

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
all_labels = []
all_predictions = []

with torch.no_grad():
    for data, labels in val_loader:
        data, labels = data.float().to(device), labels.long().to(device)
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.squeeze()).sum().item()

        # 收集标签和预测
        all_labels.extend(labels.squeeze().cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

# 计算准确率
print(f"Validation Accuracy: {100 * correct / total}%")

# 计算混淆矩阵
cm = confusion_matrix(all_labels, all_predictions)
print("混淆矩阵:")
print(cm)
