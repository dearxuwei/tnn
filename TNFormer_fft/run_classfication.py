import argparse
import os
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_imputation import Exp_Imputation
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
import random
import numpy as np

if __name__ == '__main__':
    fix_seed = 2023
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    parser = argparse.ArgumentParser(description='TimesNet')

    # 可以更改处
    params = {'tw': 500, 'Fs': 1000, 'cl': 14, 'top_k': 2, 'num_kernels': 6, 'd_model': 32,
              'e_layers': 3, 'all_runs': 700, 'subs_num': range(3,4),
              'data_path': 'E:/BaiduNetdiskDownload/mat', 'model': 'TimesNet'}

    parser.add_argument('--params', type=dict, default=params)
    # basic config
    parser.add_argument('--task_name', type=str, default='classification')
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='EEG', help='model id')
    parser.add_argument('--model', type=str, default=params['model'])

    # data loader
    parser.add_argument('--data', type=str, default='EEG', help='dataset type')
    parser.add_argument('--root_path', type=str, default=params['data_path'], help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='eeg.csv', help='data file')
    parser.add_argument('--features', type=str, default='MS',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Hourly', help='subset for M4')

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

    # model define
    parser.add_argument('--top_k', type=int, default=params['top_k'], help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=params['num_kernels'], help='for Inception')
    parser.add_argument('--enc_in', type=int, default=1, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=1, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=1, help='output size')
    parser.add_argument('--d_model', type=int, default=params['d_model'], help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=params['e_layers'], help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=params['e_layers'], help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=3, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

    # optimization
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=100, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='Exp', help='exp description')
    parser.add_argument('--loss', type=str, default='SMAPE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=False, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # subjects
    # action0 = parser.add_argument('--subjects_test', type=list, default=[1], help=argparse.SUPPRESS)
    # action1 = parser.add_argument('--subjects_train', type=list, default=[2], help=argparse.SUPPRESS)
    for eeg_len in range(params['tw'], params['tw']+1,1):
        parser.set_defaults(eeg_length=eeg_len, type=int)
        for sub_num in params['subs_num']:
            test_num = []
            test_num.append(sub_num)
            all_runs = list(range(params['all_runs']))  # 生成0到50的数字列表
            train_run = random.sample(all_runs, int(params['all_runs'] * 0.8))  # 随机选择40个作为train_runs
            test_run = [run for run in all_runs if run not in train_run]  # 剩下的就是test_runs

            parser.set_defaults(train_run=train_run, type=list)
            parser.set_defaults(test_run=test_run, type=list)
            # parser.remove_argument('--subjects_test')
            # parser._remove_action(action0)
            # parser._remove_action(action1)
            # subject = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]


            parser.set_defaults(subjects_test=test_num, type=list)
            parser.set_defaults(subjects_train=test_num, type=list)
            args = parser.parse_args()
            print('Subject ID:',args.subjects_test)
            print('Data length:',args.eeg_length)

            args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

            if args.use_gpu and args.use_multi_gpu:
                args.devices = args.devices.replace(' ', '')
                device_ids = args.devices.split(',')
                args.device_ids = [int(id_) for id_ in device_ids]
                args.gpu = args.device_ids[0]

            # print('Args in experiment:')
            # print(args)

            if args.task_name == 'long_term_forecast':
                Exp = Exp_Long_Term_Forecast
            elif args.task_name == 'short_term_forecast':
                Exp = Exp_Short_Term_Forecast
            elif args.task_name == 'imputation':
                Exp = Exp_Imputation
            elif args.task_name == 'anomaly_detection':
                Exp = Exp_Anomaly_Detection
            elif args.task_name == 'classification':
                Exp = Exp_Classification
            else:
                Exp = Exp_Long_Term_Forecast

            if args.is_training:
                for ii in range(args.itr):
                    # setting record of experiments
                    setting = '{}_{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}_{}ms'.format(
                        args.task_name,
                        args.model_id,
                        args.model,
                        args.subjects_test,
                        args.data,
                        args.features,
                        args.seq_len,
                        args.label_len,
                        args.pred_len,
                        args.d_model,
                        args.n_heads,
                        args.e_layers,
                        args.d_layers,
                        args.d_ff,
                        args.factor,
                        args.embed,
                        args.distil,
                        args.des, ii,
                        args.eeg_length)

                    exp = Exp(args)  # set experiments
                    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                    exp.train(setting)

                    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                    exp.test(setting)
                    torch.cuda.empty_cache()
            else:
                ii = 0
                setting = '{}_{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}_{}ms'.format(
                    args.task_name,
                    args.model_id,
                    args.top_k,
                    args.model,
                    args.data,
                    args.features,
                    args.seq_len,
                    args.label_len,
                    args.pred_len,
                    args.d_model,
                    args.n_heads,
                    args.e_layers,
                    args.d_layers,
                    args.d_ff,
                    args.factor,
                    args.embed,
                    args.distil,
                    args.des, ii,
                    args.eeg_length)

                exp = Exp(args)  # set experiments
                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.test(setting, test=1)
                torch.cuda.empty_cache()
