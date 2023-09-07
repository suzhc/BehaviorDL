from data_provider.data_loader import HuaweiDataset


def data_provider(args, flag):
    if args.data == 'huawei':
        data_set, data_loader = get_huawei_data(args, flag)
    else:
        raise NotImplementedError
    return data_set, data_loader