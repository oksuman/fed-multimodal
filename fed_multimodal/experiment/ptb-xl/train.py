import torch
import random
import numpy as np
import torch.nn as nn
import argparse, logging
import torch.multiprocessing
import copy, time, sys, os

from tqdm import tqdm
from pathlib import Path

from fed_multimodal.constants import constants
from fed_multimodal.trainers.server_trainer import Server
from fed_multimodal.model.mm_models import ECGClassifier
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

def parse_args():
    path_conf = dict()
    with open(str(Path(os.path.realpath(__file__)).parents[2].joinpath('system.cfg'))) as f:
        for line in f:
            key, val = line.strip().split('=')
            path_conf[key] = val.replace("\"", "")

    if path_conf["data_dir"] == ".":
        path_conf["data_dir"] = str(Path(os.path.realpath(__file__)).parents[2].joinpath('data'))
    if path_conf["output_dir"] == ".":
        path_conf["output_dir"] = str(Path(os.path.realpath(__file__)).parents[2].joinpath('output'))

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default=path_conf['output_dir'], type=str)
    parser.add_argument('--learning_rate', default=0.05, type=float)
    parser.add_argument('--global_learning_rate', default=0.05, type=float)
    parser.add_argument('--sample_rate', default=0.1, type=float)
    parser.add_argument('--num_epochs', default=300, type=int)
    parser.add_argument('--test_frequency', default=5, type=int)
    parser.add_argument('--local_epochs', default=1, type=int)
    parser.add_argument('--optimizer', default='sgd', type=str)
    parser.add_argument('--fed_alg', default='fed_avg', type=str)
    parser.add_argument('--mu', type=float, default=0.001)
    parser.add_argument('--batch_size', default=64, type=int)

    parser.add_argument('--missing_modality', action='store_true')
    parser.add_argument('--missing_modality_rate', type=float, default=0.5)

    parser.add_argument('--missing_label', action='store_true')
    parser.add_argument('--missing_label_rate', type=float, default=0.5)

    parser.add_argument('--label_nosiy', action='store_true')
    parser.add_argument('--label_nosiy_level', type=float, default=0.1)

    parser.add_argument('--att', action='store_true')
    parser.add_argument('--hid_size', type=int, default=64)
    parser.add_argument('--att_name', type=str, default='multihead')

    parser.add_argument('--modality', type=str, default='multimodal')
    parser.add_argument('--dataset', type=str, default='ptb-xl')

    parser.add_argument(
        '--method',
        type=str,
        default='concat',
        choices=['concat', 'attn', 'fedlego']
    )

    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    if args.fed_alg in ['fed_avg', 'fed_prox', 'fed_opt']:
        Client = ClientFedAvg
    elif args.fed_alg in ['scaffold']:
        Client = ClientScaffold
    else:
        Client = ClientFedRS

    save_result_dict = dict()

    for fold_idx in range(1, 6):
        dm = DataloadManager(args)
        dm.get_simulation_setting()
        dm.load_sim_dict()
        dm.get_client_ids(fold_idx=fold_idx)

        dataloader_dict = dict()
        for client_id in tqdm(dm.client_ids):
            i_avf_dict, v1_v6_dict = dm.load_ecg_feat(client_id=client_id)
            shuffle = False if client_id in ['dev', 'test'] else True
            client_sim_dict = None if client_id in ['dev', 'test'] else dm.get_client_sim_dict(client_id)
            dataloader_dict[client_id] = dm.set_dataloader(
                i_avf_dict,
                v1_v6_dict,
                shuffle=shuffle,
                client_sim_dict=client_sim_dict,
                default_feat_shape_a=np.array([1000, constants.feature_len_dict["i_to_avf"]]),
                default_feat_shape_b=np.array([1000, constants.feature_len_dict["v1_to_v6"]])
            )

        client_ids = [cid for cid in dm.client_ids if cid not in ['dev', 'test']]
        num_of_clients = len(client_ids)

        set_seed(8 * fold_idx)
        criterion = nn.BCEWithLogitsLoss().to(device)
        num_classes = constants.num_class_dict[args.dataset]

        if args.method == 'concat':
            global_model = ECGClassifier(
                num_classes=num_classes,
                i_to_avf_input_dim=constants.feature_len_dict['i_to_avf'],
                v1_to_v6_input_dim=constants.feature_len_dict['v1_to_v6'],
                en_att=False,
                d_hid=args.hid_size,
                att_name=args.att_name
            )

        elif args.method == 'attn':
            global_model = ECGClassifier(
                num_classes=num_classes,
                i_to_avf_input_dim=constants.feature_len_dict['i_to_avf'],
                v1_to_v6_input_dim=constants.feature_len_dict['v1_to_v6'],
                en_att=True,
                d_hid=args.hid_size,
                att_name=args.att_name
            )

        elif args.method == 'fedlego':
            in_shapes = [
                (1000, constants.feature_len_dict["i_to_avf"]),
                (1000, constants.feature_len_dict["v1_to_v6"])
            ]
            global_model = FedLegoClassifier(
                in_shapes=in_shapes,
                num_classes=num_classes,
                freeze_encoders=True
            )

        else:
            raise ValueError

        global_model = global_model.to(device)

        server = Server(
            args,
            global_model,
            device=device,
            criterion=criterion,
            client_ids=client_ids
        )

        server.initialize_log(fold_idx)
        server.sample_clients(num_of_clients, sample_rate=args.sample_rate)
        server.get_num_params()

        save_json_path = Path(os.path.realpath(__file__)).parents[2].joinpath(
            'result',
            args.fed_alg,
            args.dataset,
            server.feature,
            server.att,
            server.model_setting_str
        )
        Path.mkdir(save_json_path, parents=True, exist_ok=True)

        set_seed(8 * fold_idx)

        for epoch in range(int(args.num_epochs)):
            server.initialize_epoch_updates(epoch)
            skip_client_ids = []

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
                    num_class=num_classes
                )

                client.update_weights()
                server.save_train_updates(
                    copy.deepcopy(client.get_parameters()),
                    client.result['sample'],
                    client.result
                )
                del client

            server.average_weights()
            server.log_multilabel_result('train', 'macro_f')

            if epoch % args.test_frequency == 0 or epoch == int(args.num_epochs) - 1:
                with torch.no_grad():
                    server.inference(dataloader_dict['dev'])
                    server.result_dict[epoch]['dev'] = server.result
                    server.log_multilabel_result('dev', 'macro_f')

                    server.inference(dataloader_dict['test'])
                    server.result_dict[epoch]['test'] = server.result
                    server.log_multilabel_result('test', 'macro_f')

                server.log_epoch_result('macro_f')

        save_result_dict[f'fold{fold_idx}'] = server.summarize_dict_results()
        server.save_json_file(save_result_dict, save_json_path.joinpath('result.json'))

    save_result_dict['average'] = dict()
    for metric in ['macro_f', 'acc']:
        vals = []
        for k in save_result_dict:
            if metric in save_result_dict[k]:
                vals.append(save_result_dict[k][metric])
        save_result_dict['average'][metric] = np.nanmean(vals)

    server.save_json_file(save_result_dict, save_json_path.joinpath('result.json'))
