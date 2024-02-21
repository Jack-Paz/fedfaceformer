from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

import flwr as fl

import re, random, math
import numpy as np
import argparse
from tqdm import tqdm
import os, shutil
import copy

from faceformer import Faceformer
from main import trainer, test, count_parameters

from typing import Callable, Dict, List, Optional, Tuple, Union
from flwr.common import Metrics
import logging
from logging import INFO, DEBUG
from flwr.common.logger import log
import pandas as pd
from visualise_results import plot_csv_data


DEFAULT_FORMATTER = logging.Formatter(
"%(levelname)s %(name)s %(asctime)s | %(filename)s:%(lineno)d | %(message)s"
)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_train_subjects_partition(train_subjects_list, num_clients, floor=True):
    #split into equal length lists of speakers, and convert to strings with spaces in between (as expected by dataloader)
    n_subjects = len(train_subjects_list)
    partition_size = n_subjects / num_clients
    if floor==True:
        partition_size = int(partition_size) #int auto-floors
        if (partition_size < n_subjects / num_clients):
            print('WARNING, floored division on partitions, removing speakers from train set to ensure equal partition')
    else:
        raise NotImplementedError('not implemented get_train_subjects_partition without flooring yet! (equal splits only)')
    out_list = [' '.join(train_subjects_list[i:i + partition_size]) for i in range(0, n_subjects, partition_size)]
    return out_list

def lip_max_l2(vertice_dim, predict, real):
    """
    This is the lip sync metric used in the faceformer paper
    """
    from FLAMEModel.flame_masks import get_flame_mask    
    mask = get_flame_mask()
    lips_idxs = mask.lips
    lip_mask = torch.zeros((1, 5023, 3))
    lip_mask[0, lips_idxs] = 1.0
    lip_mask = lip_mask.view(1, -1)
    predict = torch.as_tensor(predict)
    real = torch.as_tensor(real)
    mask = torch.as_tensor(lip_mask)
    min_len = min([predict.shape[0], real.shape[0]])
    predict = predict[:min_len]
    real = real[:min_len]

    mask = mask.to(real.device)
    lip_pred = predict * mask
    lip_real = real * mask

    pred_verts_mm = lip_pred.view(-1, vertice_dim//3, 3) * 1000.0
    gt_verts_mm = lip_real.view(-1, vertice_dim//3, 3) * 1000.0

    diff_in_mm = pred_verts_mm - gt_verts_mm
    l2_dist_in_mm = torch.sqrt(torch.sum(diff_in_mm ** 2, dim=-1))
    max_l2_error_lip_vert, idx = torch.max(l2_dist_in_mm, dim=-1)
    mean_max_l2_error_lip_vert = torch.mean(max_l2_error_lip_vert)
    return mean_max_l2_error_lip_vert


def train(args, train_loader, model, optimizer, criterion, epoch=100):
    save_path = os.path.join(args.dataset,args.save_path)
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    iteration = 0
    for e in range(epoch):
        loss_log = []
        # train
        model.train()
        optimizer.zero_grad()
        # pbar = tqdm(enumerate(train_loader),total=len(train_loader))
        # for i, (audio, vertice, template, one_hot, file_name) in pbar:
        print(f'training epoch{e}')
        for i, (audio, vertice, template, one_hot, file_name) in enumerate(train_loader):
            iteration += 1
            audio, vertice, template, one_hot  = audio.to(device="cuda"), vertice.to(device="cuda"), template.to(device="cuda"), one_hot.to(device="cuda")
            loss = model(audio, template,  vertice, one_hot, criterion,teacher_forcing=False)
            loss.backward()
            loss_log.append(loss.item())
            if i % args.gradient_accumulation_steps==0:
                optimizer.step()
                optimizer.zero_grad()

            # pbar.set_description("(Epoch {}, iteration {}) TRAIN LOSS:{:.7f}".format((e+1), iteration ,np.mean(loss_log)))

def eval(args, dev_loader, model, criterion):
    valid_loss_log = []
    model.eval()
    train_subjects_list = [i for i in args.train_subjects.split(" ")]
    for audio, vertice, template, one_hot_all,file_name in dev_loader:
        # to gpu
        audio, vertice, template, one_hot_all= audio.to(device="cuda"), vertice.to(device="cuda"), template.to(device="cuda"), one_hot_all.to(device="cuda")
        train_subject = "_".join(file_name[0].split("_")[:-1])
        if train_subject in train_subjects_list:
            condition_subject = train_subject
            iter = train_subjects_list.index(condition_subject)
            one_hot = one_hot_all[:,iter,:]
            loss = model(audio, template,  vertice, one_hot, criterion)
            valid_loss_log.append(loss.item())
        else:
            #average across all speakers
            for iter in range(one_hot_all.shape[-1]):
                condition_subject = train_subjects_list[iter]
                one_hot = one_hot_all[:,iter,:]
                loss = model(audio, template,  vertice, one_hot, criterion)
                valid_loss_log.append(loss.item())
                    
    current_loss = np.mean(valid_loss_log)
        # print("epcoh: {}, current loss:{:.7f}".format(e+1,current_loss))    
    return current_loss

@torch.no_grad()
def test(args, model, test_loader):
    # save_path = os.path.join(args.dataset,args.save_path)
    train_subjects_list = [i for i in args.train_subjects.split(" ")]

    # model.load_state_dict(torch.load(os.path.join(save_path, '{}_model.pth'.format(epoch))))
    # model = model.to(torch.device("cuda"))
    model.eval()
    results = []
    for audio, vertice, template, one_hot_all, file_name in test_loader:
        # to gpu
        audio, vertice, template, one_hot_all= audio.to(device="cuda"), vertice.to(device="cuda"), template.to(device="cuda"), one_hot_all.to(device="cuda")
        train_subject = "_".join(file_name[0].split("_")[:-1])
        if train_subject in train_subjects_list:
            condition_subject = train_subject
            iter = train_subjects_list.index(condition_subject)
            one_hot = one_hot_all[:,iter,:]
            prediction = model.predict(audio, template, one_hot)
            prediction = prediction.squeeze() # (seq_len, V*3)
            vertice = vertice.squeeze()
            result = lip_max_l2(args.vertice_dim, prediction, vertice)
            results.append(result)
        else:
            #iterate through all conditions
            identity_results = []
            for iter in range(one_hot_all.shape[-1]):
                condition_subject = train_subjects_list[iter]
                one_hot = one_hot_all[:,iter,:]
                prediction = model.predict(audio, template, one_hot)
                prediction = prediction.squeeze() # (seq_len, V*3)
                vertice = vertice.squeeze()
                result = lip_max_l2(args.vertice_dim, prediction, vertice)
                identity_results.append(result)
            results.append(torch.min(torch.stack(identity_results)))
    return torch.mean(torch.as_tensor(results))

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

def add_to_csv(csv_path, add_dict):
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # if not set(add_dict.keys())==set(df.columns):
        #     raise Exception('cannot add new data to csv, check the dict is formatted correctly!')
    else:
        df = pd.DataFrame()
    df_to_add = pd.DataFrame(add_dict)
    merged_df = pd.concat([df, df_to_add])
    merged_df.to_csv(csv_path, index=False)
    #update graph

if __name__=='__main__':
    # taken from main.py
    import argparse
    parser = argparse.ArgumentParser(description='FaceFormer: Speech-Driven 3D Facial Animation with Transformers')
    parser.add_argument("--lr", type=float, default=0.0001, help='learning rate')
    parser.add_argument("--dataset", type=str, default="vocaset", help='vocaset or BIWI')
    parser.add_argument("--vertice_dim", type=int, default=5023*3, help='number of vertices - 5023*3 for vocaset; 23370*3 for BIWI')
    parser.add_argument("--feature_dim", type=int, default=64, help='64 for vocaset; 128 for BIWI')
    parser.add_argument("--period", type=int, default=30, help='period in PPE - 30 for vocaset; 25 for BIWI')
    parser.add_argument("--wav_path", type=str, default= "wav", help='path of the audio signals')
    parser.add_argument("--vertices_path", type=str, default="vertices_npy", help='path of the ground truth')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help='gradient accumulation')
    parser.add_argument("--max_epoch", type=int, default=20, help='number of epochs')
    parser.add_argument("--max_rounds", type=int, default=5, help='number of rounds')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--template_file", type=str, default="templates.pkl", help='path of the personalized templates')
    parser.add_argument("--save_path", type=str, default="save", help='path of the trained models')
    parser.add_argument("--result_path", type=str, default="result", help='path to the predictions')
    parser.add_argument("--train_subjects", type=str, default="FaceTalk_170728_03272_TA FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA")
    parser.add_argument("--valid_subjects", type=str, default="FaceTalk_170811_03275_TA"
       " FaceTalk_170908_03277_TA")
    parser.add_argument("--test_subjects", type=str, default="FaceTalk_170809_00138_TA"
       " FaceTalk_170731_00024_TA")
    parser.add_argument("--num_clients", type=int, default=8)
    parser.add_argument("--num_cpus", type=int, default=1)
    parser.add_argument("--num_gpus", type=float, default=1)
    parser.add_argument("--wav2vec_path", type=str, default="/home/paz/data/wav2vec2-base-960h", help='wav2vec path for the faceformer model')
    parser.add_argument("--aggr", type=str, default="avg", help='avg | mask - which aggregation method to use')
    args = parser.parse_args()

    out_dir = f'vocaset/save_federated/clients_{args.num_clients}_max_epoch_{args.max_epoch}_rounds_{args.max_rounds}_aggr_{args.aggr}'
    os.makedirs(out_dir, exist_ok=True)

    train_subjects_list = [i for i in args.train_subjects.split(" ")]
    train_subjects_list = get_train_subjects_partition(train_subjects_list, args.num_clients)

    # print("TESTING SOMETHING W TRAIN SUBJECTS LIST DELETE DELETE")
    # train_subjects_list = get_train_subjects_partition(train_subjects_list, 2)

    from data_loader import get_dataloaders


    # args.train_subjects_list = train_subjects_list #hack it into namespace for other fns
    
    #load the data only once for each client
    #WARNING: THIS LEADS TO HORRIBLE MEMORY USAGE - TRYING TO FIX
    # from data_loader import Dataset, read_data
    # train_loaders = []
    # for i, train_subjects in enumerate(train_subjects_list):
    #     print(f'loading data for client {i}')
    #     args.train_subjects = train_subjects
    #     train_data, valid_data, test_data, subjects_dict = read_data(args)
    #     train_data = Dataset(train_data,subjects_dict,"train")
    #     temp_trainloader = DataLoader(dataset=train_data, batch_size=1, shuffle=True)
    #     train_loaders.append(temp_trainloader)
    #valid and test data same for all clients
    # valid_data = Dataset(valid_data,subjects_dict,"val")
    # VALLOADER = DataLoader(dataset=valid_data, batch_size=1, shuffle=False)
    # test_data = Dataset(test_data,subjects_dict,"test")
    # testloader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)

    class FlowerClient(fl.client.NumPyClient):
        def __init__(self, net, trainloader, valloader, optimizer, criterion, cid):
            self.net = net
            self.trainloader = trainloader
            self.valloader = valloader
            self.optimizer = optimizer
            self.criterion = criterion
            self.cid = cid
            self.metrics_path = f'{out_dir}/client_{cid}_results.csv'

        def get_parameters(self, config):
            return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

        def fit(self, parameters, config):
            print(f"training client {self.cid}")
            set_parameters(self.net, parameters)
            train(args, self.trainloader, self.net, self.optimizer, self.criterion, epoch=args.max_epoch)
            #evaluate after train epochs concluded
            print(f"evaluating client {self.cid}")
            loss = eval(args, self.valloader, self.net, self.criterion)
            accuracy = test(args, self.net, self.valloader)
            # loss = eval(args, VALLOADER, self.net, self.criterion)
            # accuracy = test(args, self.net, VALLOADER)
            metrics_dict = {'loss': [loss], 'accuracy': [float(accuracy)]}
            add_to_csv(self.metrics_path, metrics_dict)
            return self.get_parameters(config={}), len(self.trainloader), {}

        def evaluate(self, parameters, config):
            #I made this a dummy function because otherwise it evaluates each client AFTER aggregation which means theyre all the same performance (because i use the same valid set for all clients)

            # log(INFO, f"evaluating client {self.cid}")
            # loss = eval(args, self.valloader, self.net, self.criterion)
            # accuracy = test(args, self.net, self.valloader)
            # metrics_dict = {'loss': [loss], 'accuracy': [float(accuracy)]}
            # add_to_csv(self.metrics_path, metrics_dict)
            print('dummy evaluate fn')
            loss = -1
            accuracy = -1 
            return float(loss), len(self.valloader), {"accuracy": float(accuracy)}

    def client_fn(cid: str) -> FlowerClient:
        """Create a Flower client representing a single organization.
        JP: This allows you to create a client x where x also corresponds to a data partition 
        """
        #splitting on subjects
        if int(cid)>len(train_subjects_list)-1:
            raise Exception('Requested n clients > n_subjects, not implemented yet')
        
        if args.aggr=='avg':
            args.train_subjects = train_subjects_list[int(cid)]
        elif args.aggr=='mask':
            pass
        else: 
            raise NotImplementedError('args.aggr chosen not implemented')
        train_subjects_subset = train_subjects_list[int(cid)]

        model = Faceformer(args)
        # to cuda
        assert torch.cuda.is_available()
        model = model.to(torch.device("cuda"))
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=args.lr)
        
        # dataset = get_dataloaders(args, return_test=False)

        dataset = get_dataloaders(args, train_subjects_subset, splits=['train','valid'])
        
        trainloader = dataset['train']
        valloader = dataset['valid']

        # Create a  single Flower client representing a single organization
        client = FlowerClient(model, trainloader, valloader, optimizer, criterion, cid).to_client()

        #eval intitial parameters - really janky implementation just for debugging
        npclient = client.numpy_client
        if not os.path.exists(npclient.metrics_path):
            print(f"evaluating client {npclient.cid} (initial params)")
            loss = eval(args, npclient.valloader, npclient.net, npclient.criterion)
            accuracy = test(args, npclient.net, npclient.valloader)
            metrics_dict = {'loss': [loss], 'accuracy': [float(accuracy)]}
            add_to_csv(npclient.metrics_path, metrics_dict)

        return client

    def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        # Multiply accuracy of each client by number of examples used
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]

        # Aggregate and return custom metric (weighted average)
        return {"accuracy": sum(accuracies) / sum(examples)}


    # The `evaluate` function will be by Flower called after every round, and once at the start of training
    def aggr_eval(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        metrics_path = f'{out_dir}/aggregated_results.csv'
        if args.aggr=='avg':
            args.train_subjects = train_subjects_list[0] #data partition 0, just used for input dim
        elif args.aggr=='mask':
            #leave args.train_subjects to initialise first layer 
            pass
        model = Faceformer(args)
        model = model.to(torch.device("cuda"))
        set_parameters(model, parameters)  # Update model with the latest parameters
        
        # dataset = get_dataloaders(args, return_test=False) #load every time for memory
        dataset = get_dataloaders(args, train_subjects_subset=None, splits=['valid'])
        valloader = dataset['valid']

        # valloader = VALLOADER #use global valloader

        criterion = nn.MSELoss()
        loss = eval(args, valloader, model, criterion)
        accuracy = test(args, model, valloader)
        
        print(f'saving aggregated metrics')
        metrics_dict = {'loss': [loss], 'accuracy': [float(accuracy)]}
        add_to_csv(metrics_path, metrics_dict)
        return loss, {"accuracy": accuracy}

    # Create FedAvg strategy

    class SaveModelStrategy(fl.server.strategy.FedAvg):
        def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
            failures: List,
        ) -> Tuple:
            """Aggregate model weights using weighted average and store checkpoint"""

            # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
            aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
            if aggregated_parameters is not None:
                if args.aggr=='avg':
                    args.train_subjects = train_subjects_list[0] #data partition 0, just used for input dim
                elif args.aggr=='mask':
                    #leave args.train_subjects to initialise first layer 
                    pass
                net = Faceformer(args)
                print(f"Saving round {server_round} aggregated_parameters...")

                # Convert `Parameters` to `List[np.ndarray]`
                aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

                # Convert `List[np.ndarray]` to PyTorch`state_dict`
                params_dict = zip(net.state_dict().keys(), aggregated_ndarrays)
                state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
                net.load_state_dict(state_dict, strict=True)

                # Save the model
                torch.save(net.state_dict(), f"{out_dir}/model_round_{server_round}.pth")
                if server_round%5!=1 and os.path.exists(f"{out_dir}/model_round_{server_round-1}.pth"):
                    os.remove(f"{out_dir}/model_round_{server_round-1}.pth") #delete old model to save space
            return aggregated_parameters, aggregated_metrics

    strategy = SaveModelStrategy(
        fraction_fit=1.0,  # Sample n% of available clients for training
        fraction_evaluate=1.0,  # Sample n% of available clients for evaluation
        min_fit_clients=args.num_clients,  # Never sample less than n clients for training
        min_evaluate_clients=args.num_clients,  # Never sample less than n clients for evaluation
        min_available_clients=args.num_clients,  # Wait until all n clients are available
        # evaluate_metrics_aggregation_fn=weighted_average,
        evaluate_fn=aggr_eval
    )

    # Specify the resources each of your clients need. By default, each
    # client will be allocated 1x CPU and 0x GPUs
    client_resources = {"num_cpus": 1, "num_gpus": 0.0}
    if DEVICE.type == "cuda":
        # here we are asigning an entire GPU for each client.
        #try num_gpus=0.5 to run 2 simulations at the same time on the gpu
        client_resources = {"num_cpus": args.num_cpus, "num_gpus": args.num_gpus}
        # Refer to our documentation for more details about Flower Simulations
        # and how to setup these `client_resources`.


    class fake_ins():
        def __init__(self):
            self.config = {}
            self.parameters = None

    if args.num_clients==1:
        print('testing with one client, doing local debug run!')
        client = client_fn('0')
        ins = fake_ins()
        best_loss = 9999
        for i in range(args.max_rounds):
            print(f'training round {i}')
            parameters = client.get_parameters(ins)
            ins.parameters = parameters.parameters
            parameters = client.fit(ins)
            ins.parameters = parameters.parameters
            print('skipping dummy eval, its all in fit')
            # result = client.evaluate(ins)
            # loss, n_examples, accuracy_dict = result.loss, result.num_examples, result.metrics
            # print(f'finished epoch {i} | loss: {loss} | accuracy: {accuracy_dict["accuracy"]}')
            # if loss < best_loss:
            #     print('new best loss, saving')
            #     best_loss = loss
            #     torch.save(client.net.state_dict(), f"{out_dir}/best_loss.pth")
            torch.save(client.numpy_client.net.state_dict(), f"{out_dir}/model_round_{i}.pth")
            # if os.path.exists(f"{out_dir}/model_round_{i-1}.pth"):
            #     os.remove(f"{out_dir}/model_round_{i-1}.pth") #delete old model to save space


    else:
        # Start simulation
        fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=args.num_clients,
            config=fl.server.ServerConfig(num_rounds=args.max_rounds),
            strategy=strategy,
            client_resources=client_resources,
        )