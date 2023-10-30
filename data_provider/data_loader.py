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
        self.all_df = self.preprocess(self.all_df)
        self.feature_df = self.all_df
        self.label_df = self.all_df[[label_flag]].dropna().reset_index().drop_duplicates().set_index('index')

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
        label = self.label_df.loc[index][self.label_flag]
        if label < 3:
            label = torch.tensor([0])
        else:
            label = torch.tensor([1])
        return data, label

    def __len__(self):
        return len(set(self.all_df.index))


class UEAloader(Dataset):
    """
    Dataset class for datasets included in:
        Time Series Classification Archive (www.timeseriesclassification.com)
    Argument:
        limit_size: float in (0, 1) for debug
    Attributes:
        all_df: (num_samples * seq_len, num_columns) dataframe indexed by integer indices, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: (num_samples * seq_len, feat_dim) dataframe; contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: (num_samples,) series of IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        labels_df: (num_samples, num_labels) pd.DataFrame of label(s) for each sample
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """

    def __init__(self, root_path, file_list=None, limit_size=None, flag=None):
        self.root_path = root_path
        self.all_df, self.labels_df = self.load_all(root_path, file_list=file_list, flag=flag)
        self.all_IDs = self.all_df.index.unique()  # all sample IDs (integer indices 0 ... num_samples-1)

        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        # use all features
        self.feature_names = self.all_df.columns
        self.feature_df = self.all_df

        # pre_process
        normalizer = Normalizer()
        self.feature_df = normalizer.normalize(self.feature_df)
        print(len(self.all_IDs))

    def load_all(self, root_path, file_list=None, flag=None):
        """
        Loads datasets from csv files contained in `root_path` into a dataframe, optionally choosing from `pattern`
        Args:
            root_path: directory containing all individual .csv files
            file_list: optionally, provide a list of file paths within `root_path` to consider.
                Otherwise, entire `root_path` contents will be used.
        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
            labels_df: dataframe containing label(s) for each sample
        """
        # Select paths for training and evaluation
        if file_list is None:
            data_paths = glob.glob(os.path.join(root_path, '*'))  # list of all paths
        else:
            data_paths = [os.path.join(root_path, p) for p in file_list]
        if len(data_paths) == 0:
            raise Exception('No files found using: {}'.format(os.path.join(root_path, '*')))
        if flag is not None:
            data_paths = list(filter(lambda x: re.search(flag, x), data_paths))
        input_paths = [p for p in data_paths if os.path.isfile(p) and p.endswith('.ts')]
        if len(input_paths) == 0:
            raise Exception("No .ts files found using pattern: '{}'".format(pattern))

        all_df, labels_df = self.load_single(input_paths[0])  # a single file contains dataset

        return all_df, labels_df

    def load_single(self, filepath):
        df, labels = load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True,
                                                             replace_missing_vals_with='NaN')
        labels = pd.Series(labels, dtype="category")
        self.class_names = labels.cat.categories
        labels_df = pd.DataFrame(labels.cat.codes,
                                 dtype=np.int8)  # int8-32 gives an error when using nn.CrossEntropyLoss

        lengths = df.applymap(
            lambda x: len(x)).values  # (num_samples, num_dimensions) array containing the length of each series

        horiz_diffs = np.abs(lengths - np.expand_dims(lengths[:, 0], -1))

        if np.sum(horiz_diffs) > 0:  # if any row (sample) has varying length across dimensions
            df = df.applymap(subsample)

        lengths = df.applymap(lambda x: len(x)).values
        vert_diffs = np.abs(lengths - np.expand_dims(lengths[0, :], 0))
        if np.sum(vert_diffs) > 0:  # if any column (dimension) has varying length across samples
            self.max_seq_len = int(np.max(lengths[:, 0]))
        else:
            self.max_seq_len = lengths[0, 0]

        # First create a (seq_len, feat_dim) dataframe for each sample, indexed by a single integer ("ID" of the sample)
        # Then concatenate into a (num_samples * seq_len, feat_dim) dataframe, with multiple rows corresponding to the
        # sample index (i.e. the same scheme as all datasets in this project)

        df = pd.concat((pd.DataFrame({col: df.loc[row, col] for col in df.columns}).reset_index(drop=True).set_index(
            pd.Series(lengths[row, 0] * [row])) for row in range(df.shape[0])), axis=0)

        # Replace NaN values
        grp = df.groupby(by=df.index)
        df = grp.transform(interpolate_missing)

        return df, labels_df

    def instance_norm(self, case):
        if self.root_path.count('EthanolConcentration') > 0:  # special process for numerical stability
            mean = case.mean(0, keepdim=True)
            case = case - mean
            stdev = torch.sqrt(torch.var(case, dim=1, keepdim=True, unbiased=False) + 1e-5)
            case /= stdev
            return case
        else:
            return case

    def __getitem__(self, ind):
        return self.instance_norm(torch.from_numpy(self.feature_df.loc[self.all_IDs[ind]].values)), \
               torch.from_numpy(self.labels_df.loc[self.all_IDs[ind]].values)

    def __len__(self):
        return len(self.all_IDs)


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