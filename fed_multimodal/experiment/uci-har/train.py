import json
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import argparse, logging
import copy, time, pickle, shutil, sys, os

from tqdm import tqdm
from pathlib import Path

from fed_multimodal.constants import constants
from fed_multimodal.trainers.server_trainer import Server
from fed_multimodal.model.mm_models import HARClassifier
from fed_multimodal.model.fedlego_model import FedLegoClassifier
from fed_multimodal.dataloader.dataload_manager import DataloadManager

from fed_multimodal.trainers.fed_rs_trainer import ClientFedRS
from fed_multimodal.trainers.fed_avg_trainer import ClientFedAvg
from fed_multimodal.trainers.scaffold_trainer import ClientScaffold

logging.basicConfig(
    format='%(asctime)s %(levelname)-3s ==> %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


class LinearUnsqueeze(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.linear(x)      # (B, out_dim)
        return x.unsqueeze(1)   # (B, 1, out_dim)

class FedLegoHARWrapper(nn.Module):
    def __init__(self, core: nn.Module):
        super().__init__()
        self.core = core

    def forward(self, x_a, x_b, l_a=None, l_b=None):
        logits = self.core([x_a, x_b])
        return logits, None

def parse_args():
    path_conf = dict()
    with open(str(Path(os.path.realpath(__file__)).parents[2].joinpath('system.cfg'))) as f:
        for line in f:
            key, val = line.strip().split('=')
            path_conf[key] = val.replace('"', '')

    if path_conf["data_dir"] == ".":
        path_conf["data_dir"] = str(Path(os.path.realpath(__file__)).parents[2].joinpath('data'))
    if path_conf["output_dir"] == ".":
        path_conf["output_dir"] = str(Path(os.path.realpath(__file__)).parents[2].joinpath('output'))

    parser = argparse.ArgumentParser(description='FedMultimoda experiments')
    parser.add_argument('--data_dir', default=path_conf["output_dir"], type=str)

    parser.add_argument('--acc_feat', default='acc', type=str)
    parser.add_argument('--gyro_feat', default='gyro', type=str)

    parser.add_argument('--learning_rate', default=0.05, type=float)
    parser.add_argument('--global_learning_rate', default=0.05, type=float)

    parser.add_argument('--sample_rate', default=0.1, type=float)
    parser.add_argument('--num_epochs', default=300, type=int)
    parser.add_argument('--test_frequency', default=1, type=int)
    parser.add_argument('--local_epochs', default=1, type=int)

    parser.add_argument('--hid_size', type=int, default=64)

    parser.add_argument('--optimizer', default='sgd', type=str)
    parser.add_argument('--mu', type=float, default=0.001)

    parser.add_argument('--fed_alg', default='fed_avg', type=str)
    parser.add_argument('--batch_size', default=16, type=int)

    parser.add_argument("--alpha", type=float, default=1.0)

    parser.add_argument('--att', type=bool, default=False)
    parser.add_argument("--en_att", dest='att', action='store_true')
    parser.add_argument('--att_name', type=str, default='multihead')

    parser.add_argument("--missing_modality", type=bool, default=False)
    parser.add_argument("--en_missing_modality", dest='missing_modality', action='store_true')

    parser.add_argument("--missing_modailty_rate", type=float, default=0.5)

    parser.add_argument("--missing_label", type=bool, default=False)
    parser.add_argument("--en_missing_label", dest='missing_label', action='store_true')
    parser.add_argument("--missing_label_rate", type=float, default=0.5)

    parser.add_argument('--label_nosiy', type=bool, default=False)
    parser.add_argument("--en_label_nosiy", dest='label_nosiy', action='store_true')
    parser.add_argument('--label_nosiy_level', type=float, default=0.1)

    parser.add_argument("--dataset", type=str, default="uci-har")
    parser.add_argument('--modality', type=str, default='multimodal')

    parser.add_argument(
        '--method',
        type=str,
        default='concat',
        choices=['concat', 'attn', 'fedlego']
    )

    args = parser.parse_args()

    # dataload_manager가 args.missing_modality_rate를 참조하는 코드가 있어 typo alias를 맞춰줌
    if not hasattr(args, "missing_modality_rate"):
        args.missing_modality_rate = args.missing_modailty_rate

    return args

if __name__ == '__main__':

    args = parse_args()

    dm = DataloadManager(args)
    dm.get_simulation_setting(alpha=args.alpha)

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        print('GPU available, use GPU')

    save_result_dict = dict()

    if args.fed_alg in ['fed_avg', 'fed_prox', 'fed_opt']:
        Client = ClientFedAvg
    elif args.fed_alg in ['scaffold']:
        Client = ClientScaffold
    elif args.fed_alg in ['fed_rs']:
        Client = ClientFedRS
    else:
        Client = ClientFedAvg

    dm.load_sim_dict()
    dm.get_client_ids()

    dataloader_dict = dict()
    logging.info('Reading Data')
    for client_id in tqdm(dm.client_ids):
        acc_dict = dm.load_acc_feat(client_id=client_id)
        gyro_dict = dm.load_gyro_feat(client_id=client_id)
        shuffle = False if client_id in ['dev', 'test'] else True
        client_sim_dict = None if client_id in ['dev', 'test'] else dm.get_client_sim_dict(client_id=client_id)
        dataloader_dict[client_id] = dm.set_dataloader(
            acc_dict,
            gyro_dict,
            shuffle=shuffle,
            client_sim_dict=client_sim_dict,
            default_feat_shape_a=np.array([128, constants.feature_len_dict[args.acc_feat]]),
            default_feat_shape_b=np.array([128, constants.feature_len_dict[args.gyro_feat]]),
        )

    for fold_idx in range(1, 6):
        client_ids = [client_id for client_id in dm.client_ids if client_id not in ['dev', 'test']]
        num_of_clients = len(client_ids)

        set_seed(8 * fold_idx)

        criterion = nn.NLLLoss().to(device)

        if args.method == 'fedlego':
            acc_in = 128 * constants.feature_len_dict[args.acc_feat]
            gyro_in = 128 * constants.feature_len_dict[args.gyro_feat]
            zdim = 64

            encoders = [
                nn.Sequential(
                    nn.Flatten(),
                    LinearUnsqueeze(acc_in, zdim),
                ),
                nn.Sequential(
                    nn.Flatten(),
                    LinearUnsqueeze(gyro_in, zdim),
                )
            ]

            core = FedLegoClassifier(
                in_shapes=[
                    (128, constants.feature_len_dict[args.acc_feat]),
                    (128, constants.feature_len_dict[args.gyro_feat])
                ],
                num_classes=constants.num_class_dict[args.dataset],
                encoders=encoders,
                freeze_encoders=False
            )

            global_model = FedLegoHARWrapper(core)


        else:
            global_model = HARClassifier(
                num_classes=constants.num_class_dict[args.dataset],
                acc_input_dim=constants.feature_len_dict[args.acc_feat],
                gyro_input_dim=constants.feature_len_dict[args.gyro_feat],
                en_att=(args.method == 'attn'),
                d_hid=args.hid_size,
                att_name=args.att_name
            )

        global_model = global_model.to(device)

        server = Server(
            args,
            global_model,
            device=device,
            criterion=criterion,
            client_ids=client_ids
        )
        server.initialize_log(fold_idx)
        server.sample_clients(
            num_of_clients,
            sample_rate=args.sample_rate
        )
        server.get_num_params()

        save_json_path = Path(os.path.realpath(__file__)).parents[2].joinpath(
            'result',
            args.fed_alg,
            args.dataset,
            server.feature,
            server.att,
            server.model_setting_str,
            args.method
        )
        Path.mkdir(save_json_path, parents=True, exist_ok=True)

        set_seed(8 * fold_idx)

        for epoch in range(int(args.num_epochs)):
            server.initialize_epoch_updates(epoch)
            skip_client_ids = list()

            for idx in server.clients_list[epoch]:
                client_id = client_ids[idx]
                dataloader = dataloader_dict[client_id]
                if dataloader is None:
                    skip_client_ids.append(client_id)
                    continue

                client = Client(
                    args,
                    device,
                    criterion,
                    dataloader,
                    model=copy.deepcopy(server.global_model),
                    num_class=constants.num_class_dict[args.dataset]
                )

                client.update_weights()
                server.save_train_updates(
                    copy.deepcopy(client.get_parameters()),
                    client.result['sample'],
                    client.result
                )
                del client

            logging.info(f'Client Round: {epoch}, Skip client {skip_client_ids}')

            if len(server.num_samples_list) == 0:
                continue

            server.average_weights()
            logging.info('---------------------------------------------------------')
            server.log_classification_result(data_split='train', metric='f1')

            if epoch % args.test_frequency == 0:
                with torch.no_grad():
                    server.inference(dataloader_dict['dev'])
                    server.result_dict[epoch]['dev'] = server.result
                    server.log_classification_result(data_split='dev', metric='f1')

                    server.inference(dataloader_dict['test'])
                    server.result_dict[epoch]['test'] = server.result
                    server.log_classification_result(data_split='test', metric='f1')

                logging.info('---------------------------------------------------------')
                server.log_epoch_result(metric='f1')
            logging.info('---------------------------------------------------------')

        save_result_dict[f'fold{fold_idx}'] = server.summarize_dict_results()

        server.save_json_file(
            save_result_dict,
            save_json_path.joinpath('result.json')
        )

    save_result_dict['average'] = dict()
    for metric in ['f1', 'acc', 'top5_acc']:
        result_list = list()
        for key in save_result_dict:
            if metric not in save_result_dict[key]:
                continue
            result_list.append(save_result_dict[key][metric])
        save_result_dict['average'][metric] = np.nanmean(result_list)

    server.save_json_file(
        save_result_dict,
        save_json_path.joinpath('result.json')
    )
