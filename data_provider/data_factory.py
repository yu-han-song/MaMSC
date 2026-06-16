from data_provider.data_loader import Dataset_Ninapro
from torch.utils.data import DataLoader

data_dict = {
    'ninapro': Dataset_Ninapro,
}

def data_provider(args, flag):

    Data = data_dict[args.data]
    shuffle_flag = False if flag in ['test', 'TEST'] else True
    drop_last = False
    batch_size = args.batch_size

    data_set = Data(
        args=args,
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag
    )
    print(f'{flag} set size:', len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader