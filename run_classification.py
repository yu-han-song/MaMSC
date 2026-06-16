import argparse
import os
import torch
from exp.exp_classification import Exp_Classification
import random
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MaMSC for sEMG Classification')
    parser.add_argument('--lradj', type=str, default='fixed', help='adjust learning rate')
    parser.add_argument('--random_seed', type=int, default=2024)
    parser.add_argument('--is_training', type=int, required=True, default=1)
    parser.add_argument('--model_id', type=str, required=True, default='Mamsc')
    parser.add_argument('--model', type=str, default='Mamsc')
    parser.add_argument('--task_name', type=str, default='classification')

    parser.add_argument('--data', type=str, default='ninapro')
    parser.add_argument('--root_path', type=str, default='')
    parser.add_argument('--data_path', type=str, default='')

    parser.add_argument('--seq_len', type=int, default=400)
    parser.add_argument('--enc_in', type=int, default=12)
    parser.add_argument('--c_out', type=int, default=17)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1)

    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--expand', type=int, default=2)
    parser.add_argument('--d_state', type=int, default=32)
    parser.add_argument('--d_conv', type=int, default=4)
    parser.add_argument('--embed_dim', type=int, default=32)

    parser.add_argument('--train_epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--itr', type=int, default=1)
    parser.add_argument('--des', type=str, default='test')

    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--use_multi_gpu', action='store_true', default=False)
    parser.add_argument('--devices', type=str, default='0,1')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

    args = parser.parse_args()

    args.use_gpu = torch.cuda.is_available() and args.use_gpu
    if args.use_gpu and args.use_multi_gpu:
        device_ids = [int(i) for i in args.devices.split(',')]
        args.device_ids = device_ids
        args.gpu = device_ids[0]

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)

    print('Args in experiment:')
    print(args)

    setting = '{}_{}_sl{}_bs{}_lr{}'.format(
        args.model_id,
        args.data,
        args.seq_len,
        args.batch_size,
        args.learning_rate
    )
    exp = Exp_Classification(args)

    def safe_load_weights(model, path, device, is_strict=True):
        print(f'>>>>>>> Loading Weights from: {path} >>>>>>>')
        dummy_input = torch.randn(1, args.seq_len, args.enc_in).to(device)
        model.eval()
        with torch.no_grad():
            _ = model(dummy_input)
        state_dict = torch.load(path, map_location='cpu')
        model.load_state_dict(state_dict, strict=is_strict)


    if args.is_training:
        print('>>>>>>> Start Training >>>>>>>')
        exp.train(setting)

        print('>>>>>>> Start Testing >>>>>>>')
        exp.test(setting)

    else:
        print('>>>>>>> Start Testing (Zero-shot) >>>>>>>')
        exp.test(setting, test=1)



