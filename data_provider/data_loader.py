import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from data_provider.uea import subsample, interpolate_missing, Normalizer
from sktime.datasets import load_from_tsfile_to_dataframe
from imblearn.under_sampling import RandomUnderSampler
import warnings

warnings.filterwarnings('ignore')


class HuaweiDataset(Dataset):
    """
    Dataset class for datasets included in:
        Time Series Classification Archive (www.timeseriesclassification.com)
    Argument:
        limit_size: float in (0, 1) for debug
    Attributes:
        all_df: (num_samples * seq_len, num_columns) dataframe indexed by integer indices, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: (num_samples * seq_len, feat_dim) dataframe; contains the subset of columns of `all_df` which correspond to selected features
        labels_df: (num_samples, num_labels) pd.DataFrame of label(s) for each sample
    """
    def __init__(self, path, label_flag='emotion'):
        '''
        label_flag: 'emotion', 'energy'
        '''
        self.path = path
        self.label_flag = label_flag

        bloodoxygen = self.read_bloodoxygen()
        heartrate = self.read_heartrate()
        activity = self.read_activity()
        df = activity.merge(bloodoxygen, on=['externalid', 'recordtime', 'day'], how='outer')\
            .merge(heartrate, on=['externalid', 'recordtime', 'day'], how='outer').set_index(['externalid', 'day'])
        # 关于标签的处理，在read_ema()函数中进行（如删除分数在3-5之间的数据实例）
        self.ema = self.read_ema()
        self.all_df = pd.merge(df, self.ema, left_index=True, right_index=True, how='inner')
        # 对数据进行预处理
        self.all_df = self.preprocess(self.all_df)
        # 根据标签进行采样
        # self.all_df = self.resample(self.all_df)
        self.feature_df = self.all_df
        self.labels_df = self.all_df[[label_flag]].dropna().reset_index().drop_duplicates().set_index('index')

    def preprocess(self, df):

        df.activityName = df.activityName.fillna('unknown').astype('category').cat.codes
        
        # 使用nan填充，保证每个用户每天都有每一分钟的数据
        df_list = []
        for id, grp_id in df.groupby('externalid'):
            for day, grp_id_day in grp_id.groupby('day'):
                
                grp_id_day = grp_id_day.set_index('recordtime').resample('T').first().reset_index()

                full_day_range = pd.date_range(start=day, \
                                    end=day + pd.Timedelta(days=1) - pd.Timedelta(minutes=1), freq='T')
                full_day_df = pd.DataFrame({'recordtime': full_day_range})
                full_day_df = pd.merge(full_day_df, grp_id_day, on='recordtime', how='left')
                full_day_df['externalid'] = id
                full_day_df['day'] = day

                # 填充emotion和energy
                emotion = full_day_df['emotion'].mode().values[0]
                energy = full_day_df['energy'].mode().values[0]
                full_day_df['emotion'] = full_day_df['emotion'].fillna(emotion)
                full_day_df['energy'] = full_day_df['energy'].fillna(energy)
                
                df_list.append(full_day_df)
        
        df = pd.concat(df_list).set_index(['externalid', 'day'])

        df['index'], _ = pd.factorize(df.index)
        df.set_index('index', inplace=True)

        return df
    
    def resample(self, df):
        # 根据标签进行采样
        # 使用简单的方法进行下采样
        rus = RandomUnderSampler(random_state=0)
        # 创建一个采样df
        resample = df[self.label_flag].reset_index().drop_duplicates()
        resample[self.label_flag] = resample[self.label_flag].apply(lambda x: 0 if x < 3 else 1)
        resample, _ = rus.fit_resample(resample, resample.emotion)
        resample = resample.set_index('index')
        # 根据采样df对原始df进行采样
        df = df.loc[resample.index]
        # 根据原index的唯一值重新映射index
        # 获取原始索引的唯一值，并按照出现顺序进行排序
        unique_indices = sorted(set(df.index))
        # 创建映射字典
        index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_indices)}
        # 使用 map() 函数映射索引
        df.index = df.index.map(index_mapping)
        return df

    def read_activity(self):
        file_path = os.path.join(self.path, 'activity.pkl.zip')
        df = pd.read_pickle(file_path)[['externalid', 'recordtime', 'activityName', 'step']]
        df['recordtime'] = pd.to_datetime(df['recordtime'], unit='ms') + pd.Timedelta(hours=8)
        df['day'] = df['recordtime'].dt.date
        return df
    
    def read_bloodoxygen(self):
        file_path = os.path.join(self.path, 'bloodoxygen.pkl.zip')
        df = pd.read_pickle(file_path)[['externalid', 'recordtime', 'avgOxygenSaturation']]
        df['recordtime'] = pd.to_datetime(df['recordtime'], unit='ms') + pd.Timedelta(hours=8)
        df['day'] = df['recordtime'].dt.date
        return df
    
    def read_heartrate(self):
        file_path = os.path.join(self.path, 'heartrate.pkl.zip')
        df = pd.read_pickle(file_path)[['externalid', 'recordtime', 'avgHeartRate']]
        df['recordtime'] = pd.to_datetime(df['recordtime'], unit='ms') + pd.Timedelta(hours=8)
        df['day'] = df['recordtime'].dt.date
        return df
    
    def read_ema(self):
        df = pd.read_csv(os.path.join(self.path, 'ema.csv'))
        df.columns = ['externalid','用户编号','姓名','emotion','energy','sleep_time','wake_time','recordtime']
        df = df[['externalid','emotion','energy','recordtime']]
        df.dropna(inplace=True)
        df['externalid'] = df['externalid'].astype(int)
        df['recordtime'] = pd.to_datetime(df['recordtime'])
        df['day'] = df['recordtime'].dt.date
        return df.groupby(['externalid','day']).mean().query(f'{self.label_flag} < 3 or {self.label_flag} > 5')

    def __getitem__(self, index):
        data = torch.from_numpy(self.all_df.loc[index][['activityName', 'step', \
                                                 'avgOxygenSaturation', 'avgHeartRate']].fillna(0).values)
        label = self.labels_df.loc[index][self.label_flag]
        if label < 3:
            label = torch.tensor([0])
        else:
            label = torch.tensor([1])
        return data, label

    def __len__(self):
        return len(set(self.all_df.index))
    
    def __iter__(self):
        self._current = 0
        return self

    def __next__(self):
        if self._current >= self.__len__():
            raise StopIteration
        else:
            result = self.__getitem__(self._current)
            self._current += 1
            return result


def padding_mask(lengths, max_len=None):
    """
    Used to mask padded positions: creates a (batch_size, max_len) boolean mask from a tensor of sequence lengths,
    where 1 means keep element at this position (time step)
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max_val()  # trick works because of overloading of 'or' operator for non-boolean types
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))

def collate_fn(data, max_len=None):
    """Build mini-batch tensors from a list of (X, mask) tuples. Mask input. Create
    Args:
        data: len(batch_size) list of tuples (X, y).
            - X: torch tensor of shape (seq_length, feat_dim); variable seq_length.
            - y: torch tensor of shape (num_labels,) : class indices or numerical targets
                (for classification or regression, respectively). num_labels > 1 for multi-task models
        max_len: global fixed sequence length. Used for architectures requiring fixed length input,
            where the batch length cannot vary dynamically. Longer sequences are clipped, shorter are padded with 0s
    Returns:
        X: (batch_size, padded_length, feat_dim) torch tensor of masked features (input)
        targets: (batch_size, padded_length, feat_dim) torch tensor of unmasked features (output)
        target_masks: (batch_size, padded_length, feat_dim) boolean torch tensor
            0 indicates masked values to be predicted, 1 indicates unaffected/"active" feature values
        padding_masks: (batch_size, padded_length) boolean tensor, 1 means keep vector at this position, 0 means padding
    """

    batch_size = len(data)
    features, labels = zip(*data)

    # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
    lengths = [X.shape[0] for X in features]  # original sequence length for each time series
    if max_len is None:
        max_len = max(lengths)

    X = torch.zeros(batch_size, max_len, features[0].shape[-1])  # (batch_size, padded_length, feat_dim)
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        X[i, :end, :] = features[i][:end, :]

    targets = torch.stack(labels, dim=0)  # (batch_size, num_labels)

    padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16),
                                 max_len=max_len)  # (batch_size, padded_length) boolean tensor, "1" means keep

    return X, targets, padding_masks


def data_provider(path, label_flag='emotion'):
    data_set = HuaweiDataset(path, label_flag)
    data_loader = DataLoader(
        data_set, 
        batch_size=20, 
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, max_len=1500)
    )
    return data_set, data_loader

def data_loader(dataset, flag):

    if flag == 'train':
        shuffle = True
    else:
        shuffle = False

    data_loader = DataLoader(
        dataset,
        batch_size=20,
        shuffle=shuffle,
        collate_fn=lambda x: collate_fn(x, max_len=1500)
    )
    return data_loader