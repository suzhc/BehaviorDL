from data_provider.data_loader import HuaweiDataset, UEAloader, DataLoader
from data_provider.uea import collate_fn

from torch.utils.data import random_split
from models import PatchTST
from torch import optim
from torch import nn

import numpy as np
import time

from dataclasses import dataclass
@dataclass
class args:
    data_path: str
    label_flag: str
    epochs: int
    seq_len: int
    d_model: int
    dropout: float
    e_layers: int
    d_ff: int
    output_attention: bool
    num_class: int = 2
    activation: str = 'gelu'
    factor: float = 1
    n_heads: int = 8
    enc_in: int = 7
    pred_len: int = 0
args.data_path = '/home/user/suzhao/BehaviorDL/dataset/Huawei'
args.label_flag = 'emotion'
args.epochs = 10
args.seq_len = 1441
args.d_model = 128
args.dropout = 0.6
args.e_layers = 3
args.d_ff = 128
args.output_attention = False
args.activation = 'gelu'
args.factor = 1
args.n_heads = 8
args.num_class = 2
args.enc_in = 7


data_set = UEAloader('/home/user/suzhao/BehaviorDL/dataset/Heartbeat')

# train_size = int(0.8 * len(data_set))
# test_size = len(data_set) - train_size
# train_dataset, test_dataset = random_split(data_set, [train_size, test_size])

# train_loader = DataLoader(
#     train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
# test_loader = DataLoader(
#     test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

train_loader = DataLoader(
    data_set, batch_size=16, shuffle=True, collate_fn=collate_fn)

def get_data(flag='TRAIN'):
    if flag == 'TRAIN':
        train_dataset = UEAloader('/home/user/suzhao/BehaviorDL/dataset/Heartbeat', flag='TRAIN')
        train_loader = DataLoader(
            train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
        return train_dataset, train_loader
    elif flag == 'TEST':
        test_dataset = UEAloader('/home/user/suzhao/BehaviorDL/dataset/Heartbeat', flag='TEST')
        test_loader = DataLoader(
            test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
        return test_dataset, test_loader
    else:
        raise NotImplementedError

def build_model():
    # model input depends on data
    train_data, train_loader = get_data(flag='TRAIN')
    test_data, test_loader = get_data(flag='TEST')
    args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)
    args.pred_len = 0
    args.enc_in = train_data.feature_df.shape[1]
    args.num_class = len(train_data.class_names)
    # model init
    model = PatchTST.Model(args).float()
    return model

model = build_model()

train_steps = len(train_loader)
model_optim = optim.Adam(model.parameters(), lr=0.02)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    iter_count = 0
    train_loss = []

    model.train()
    epoch_time = time.time()

    for i, (batch_x, label, padding_mask) in enumerate(train_loader):
        model_optim.zero_grad()

        batch_x = batch_x.float()
        padding_mask = padding_mask.float()

        output = model(batch_x, padding_mask)
        loss = criterion(output, label.long().squeeze(-1))

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
        model_optim.step()
    
    print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))