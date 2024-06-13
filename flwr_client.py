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

from main import trainer, test, count_parameters

from typing import Callable, Dict, List, Optional, Tuple, Union
from flwr.common import Metrics
import logging
from logging import INFO, DEBUG
from flwr.common.logger import log
import pandas as pd
from visualise_results import plot_csv_data

from imitator.utils.losses import Custom_errors

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

def compute_auxillary_losses(predicted_mesh, gt_mesh, custom_loss):
    velocity_loss, velocity_loss_weighted = custom_loss.velocity_loss(predicted_mesh, gt_mesh)  # predict, real
    aux_loss_sum = velocity_loss_weighted
    loss_dict = {
        "aux_losses": aux_loss_sum,
        "velocity_loss": velocity_loss,
    }
    return loss_dict

def accumulate_and_noise(accumulated_grads, noise_scale=1.0):
    noised_grads = []
    for grad_list in accumulated_grads:
        if not grad_list:
            noised_grads.append(None) #NOT SURE IF ZERO IS APPROPRIATE?
            continue
        # Stack gradients along a new dimension, compute mean across this dimension
        stacked_grads = torch.stack(grad_list)
        mean_grad = torch.mean(stacked_grads, dim=0)

        # Add Gaussian noise
        noise = torch.randn_like(mean_grad) * noise_scale
        noised_grads.append(mean_grad + noise)

    return noised_grads

def calculate_epsilon(steps, sigma, sensitivity, delta):
    return (sensitivity / sigma) * np.sqrt(2 * np.log(1.25 / delta)) * np.sqrt(steps)


def train(args, train_loader, model, optimizer, criterion, custom_loss, accountant, fileid_to_filename, epoch=100):
    iteration = 0
    epoch_losses = []
    for e in range(epoch):
        loss_log = []
        # train
        model.train()
        optimizer.zero_grad()
        # pbar = tqdm(enumerate(train_loader),total=len(train_loader))
        # for i, (audio, vertice, template, one_hot, file_name) in pbar:
        print(f'training epoch{e}')
        for audio, vertice, template, one_hot, fileid in train_loader:
            if len(fileid)==0:
                file_name = ''
            else:
                fileid = int(fileid[0][0]) #if batched just take the first speaker for conditioning?
                file_name = fileid_to_filename[fileid]
            iteration += 1
            audio, vertice, template, one_hot  = audio.to(device="cuda"), vertice.to(device="cuda"), template.to(device="cuda"), one_hot.to(device="cuda")
            if args.model=='faceformer':
                loss = model(audio, template,  vertice, one_hot, criterion,teacher_forcing=False)
            elif args.model=='imitator_gen':
                #train generalised model, use normal loss
                loss, vertice_out_w_style = model(audio, template,  vertice, one_hot, criterion,teacher_forcing=False)
            elif args.model.startswith('imitator_stg'):
                #train speaker-specific model, use style loss
                if type(file_name)==list:
                    assert len(file_name)==1
                    file_name = file_name[0]
                rec_loss, pred_verts = model.style_forward(audio, file_name, template,  vertice, one_hot, criterion,teacher_forcing=False)
                subjsen = file_name[0].split(".")[0]
                mbp_loss = custom_loss.compute_mbp_reconstruction_loss(pred_verts, vertice, subjsen)
                aux_loss = compute_auxillary_losses(pred_verts, vertice, custom_loss)
                loss = rec_loss + aux_loss["aux_losses"] + mbp_loss
                #rec_loss = L_mse
                #aux_loss = velocity loss, weighted
                #mpb_loss = lip closure (m, b, p = bilabial)

            if loss.requires_grad==False:
                continue #possibly causing an issue
            loss.backward()
            loss_log.append(loss.item())
            
            #testing to find which module is not updated in backward
            if args.dp != 'none':
                counter = 0
                for pid, (name, p) in enumerate(model.named_parameters()):
                    # print(name, pid, p.requires_grad, p.shape)
                    if not p.requires_grad:
                        # print('doesnt require grad')
                        continue
                    # breakpoint()
                    p.grad_sample = p.grad.unsqueeze(0)
                    # SETTING GRAD SAMPLE TO GRAD BECAUSE bsz = 1

                    # per_sample_grad = p.grad_sample
                    # counter += 1
                    # if per_sample_grad is None:
                    #     print(f'got none grad')
                    #     if (p.grad.sum()==0).item()==True:
                    #         p.grad_sample = p.grad.unsqueeze(0).repeat(vertice.shape[0], 1, 1)
                    #         # p.grad_sample = p.grad.unsqueeze(0).repeat(69, 1, 1)
                    #         print('new grad shape', p.grad_sample.shape)
                    # else:
                    #     print('grad shape', p.grad_sample.shape)
                    # breakpoint()
                # grad_samples = [x.grad_sample for x in model.parameters() if x.requires_grad]
    
            if args.dp=='opacus':
                optimizer.step()
                optimizer.zero_grad()
                epsilon = accountant.get_epsilon(delta=args.delta)
                print(f"(ε = {epsilon:.2f}, δ = {args.delta})")
            elif args.dp=='basic':
                accumulated_grads = [[] for _ in model.parameters()]  # List of lists for each parameter
                for pid, p in enumerate(model.parameters()):
                    if (p.grad is not None) and (len(p.grad)>0):
                        per_sample_grad = p.grad.detach().clone()
                        torch.nn.utils.clip_grad_norm_(per_sample_grad, max_norm=1.0)
                        accumulated_grads[pid].append(per_sample_grad)
                model.zero_grad()
                noise_scale = 1
                aggregated_and_noised_grads = accumulate_and_noise(accumulated_grads, noise_scale=noise_scale)
                for param, agg_grad in zip(model.parameters(), aggregated_and_noised_grads):
                    param.grad = agg_grad
                    
                sigma = 1.0  # Standard deviation of the Gaussian noise
                sensitivity = 1.0  # Assuming a clipping norm of 1
                delta = 1e-3  # Acceptable failure probability of the privacy guarantee
                epsilon = calculate_epsilon(iteration, sigma, sensitivity, delta)
                print(f"Approximate ε after {iteration} steps: {epsilon}")
            
            if not args.dp=='opacus':
                if iteration % args.gradient_accumulation_steps==0:
                    optimizer.step()
                    optimizer.zero_grad()

        epoch_loss = np.mean(loss_log)
        epoch_losses.append(epoch_loss)
    return epoch_losses

def eval(args, dev_loader, model, criterion, custom_loss, fileid_to_filename):
    valid_loss_log = []
    model.eval()
    train_subjects_list = [i for i in args.all_train_subjects.split(" ")]
    for audio, vertice, template, one_hot_all,fileid in dev_loader:
        fileid = int(fileid)
        file_name = fileid_to_filename[fileid]
        # to gpu
        audio, vertice, template, one_hot_all= audio.to(device="cuda"), vertice.to(device="cuda"), template.to(device="cuda"), one_hot_all.to(device="cuda")
        train_subject = "_".join(file_name[0].split("_")[:-1])
        if train_subject in train_subjects_list:
            train_subject_id = train_subjects_list.index(condition_subject)
            train_subject_ids = [train_subject_id]
            #we saw this subject in training, condition on them
        else:
            train_subject_ids = range(one_hot_all.shape[-1])
            #subject not in training, take the average of all subjects

        for iter in train_subject_ids:
            condition_subject = train_subjects_list[iter]
            one_hot = one_hot_all[:,iter,:]
            if args.model=='faceformer':
                loss = model(audio, template,  vertice, one_hot, criterion)
            elif args.model=='imitator_gen':
                #train generalised model, use normal loss
                loss, vertice_out_w_style = model(audio, template,  vertice, one_hot, criterion,teacher_forcing=False)
            elif args.model.startswith('imitator_stg'):
                #train speaker-specific model, use style loss
                if type(file_name)==list:
                    assert len(file_name)==1
                    file_name = file_name[0]
                rec_loss, pred_verts = model.style_forward(audio, file_name, template,  vertice, one_hot, criterion,teacher_forcing=False)
                subjsen = file_name[0].split(".")[0]
                mbp_loss = custom_loss.compute_mbp_reconstruction_loss(pred_verts, vertice, subjsen)
                aux_loss = compute_auxillary_losses(pred_verts, vertice, custom_loss)
                loss = rec_loss + aux_loss["aux_losses"] + mbp_loss
        valid_loss_log.append(loss.item())                    
    current_loss = np.mean(valid_loss_log)
        # print("epcoh: {}, current loss:{:.7f}".format(e+1,current_loss))    
    return current_loss

@torch.no_grad()
def test(args, model, test_loader, fileid_to_filename):
    # save_path = os.path.join(args.dataset,args.save_path)
    train_subjects_list = [i for i in args.all_train_subjects.split(" ")]

    # model.load_state_dict(torch.load(os.path.join(save_path, '{}_model.pth'.format(epoch))))
    # model = model.to(torch.device("cuda"))
    model.eval()
    results = []
    for audio, vertice, template, one_hot_all, fileid in test_loader:
        # to gpu
        fileid = int(fileid[0]) #just take the first speaker for conditioning?
        file_name = fileid_to_filename[fileid]
        audio, vertice, template, one_hot_all= audio.to(device="cuda"), vertice.to(device="cuda"), template.to(device="cuda"), one_hot_all.to(device="cuda")
        train_subject = "_".join(file_name[0].split("_")[:-1])
        if train_subject in train_subjects_list:
            train_subject_id = train_subjects_list.index(condition_subject)
            train_subject_ids = [train_subject_id]
            #we saw this subject in training, condition on them
        else:
            train_subject_ids = range(one_hot_all.shape[-1])
            #subject not in training, take the best result across all subjects (like Imitator does)
        
        identity_results = []
        for iter in train_subject_ids:
            condition_subject = train_subjects_list[iter]
            one_hot = one_hot_all[:,iter,:]
            prediction = model(audio, template, vertice=None, one_hot=one_hot, criterion=None, test_dataset=test_loader)
            prediction = prediction.squeeze() # (seq_len, V*3)
            vertice = vertice.squeeze()
            result = lip_max_l2(args.vertice_dim, prediction, vertice)
            identity_results.append(result)
        results.append(torch.min(torch.stack(identity_results)))
        # else:
        #     #iterate through all conditions
        #     print('iterating all test conditions')
        #     identity_results = []
        #     for iter in range(one_hot_all.shape[-1]):
        #         condition_subject = train_subjects_list[iter]
        #         one_hot = one_hot_all[:,iter,:]
        #         prediction = model.predict(audio, template, one_hot)
        #         prediction = prediction.squeeze() # (seq_len, V*3)
        #         vertice = vertice.squeeze()
        #         result = lip_max_l2(args.vertice_dim, prediction, vertice)
        #         identity_results.append(result)
        #     results.append(torch.min(torch.stack(identity_results)))
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


def freeze_p1(model):
    print("\nFreezing from the imitator_org_vert_reg_style_optim\n")
    # freeze the model expect the style emebeddding
    for param in model.parameters():
        param.requires_grad = False

    # unfrezzing the style encoder
    for param in model.obj_vector.parameters():
        param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('trainable params after freezing p1:', trainable_params)
    return model

def freeze_p2(model):
    # freeze the model expect the style emebeddding
    for param in model.parameters():
        param.requires_grad = False

    # unfreezing the style encoder
    for param in model.obj_vector.parameters():
        param.requires_grad = True

    # unfreeze the motion decoder
    # v
    for name, param in model.vertice_map_r.named_parameters():
        if "final_out_layer" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of trainiable params after freezing p2", trainable_params)
    return model


if __name__=='__main__':
    # taken from main.py
    import argparse
    parser = argparse.ArgumentParser(description='FaceFormer: Speech-Driven 3D Facial Animation with Transformers')
    parser.add_argument("--lr", type=float, default=0.0001, help='learning rate')
    parser.add_argument("--dir", type=str, default='.', help='path to working dir')
    parser.add_argument("--dataset", type=str, default="vocaset", help='vocaset or BIWI | hdtf | combined (hdtf + vocaset)')
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
    parser.add_argument("--save_path", type=str, default="save_federated", help='path of the trained models')
    parser.add_argument("--result_path", type=str, default="result", help='path to the predictions')
    parser.add_argument("--train_subjects", type=str, default="FaceTalk_170728_03272_TA FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA")
    parser.add_argument("--all_train_subjects", type=str, default="FaceTalk_170728_03272_TA FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA")

    parser.add_argument("--valid_subjects", type=str, default="FaceTalk_170811_03275_TA"
       " FaceTalk_170908_03277_TA")
    parser.add_argument("--test_subjects", type=str, default="FaceTalk_170809_00138_TA"
       " FaceTalk_170731_00024_TA")
    parser.add_argument("--num_clients", type=int, default=8)
    parser.add_argument("--num_cpus", type=int, default=1)
    parser.add_argument("--num_gpus", type=float, default=1)

    parser.add_argument("--wav2vec_path", type=str, default="/home/paz/data/wav2vec2-base-960h", help='wav2vec path for the faceformer model')
    parser.add_argument("--aggr", type=str, default="avg", help='avg | mask - which aggregation method to use')
    parser.add_argument("--dp", type=str, default="none", help='none | fixed | adaptive | opacus | basic - which differential privacy method to use')
    parser.add_argument("--data_split", type=str, default="vertical", help='vertical | horziontal - vertical=split on speakers, horzontal=split some of each train speaker for test and valid')
    parser.add_argument("--train_idx", type=int, default=-1, help='index of speaker to train on for individual run, -1 = train on all speakers')
    parser.add_argument("--condition_idx", type=int, default=2, help='for imitator runs, which speaker to initialise the personalisation layer')
    parser.add_argument("--model", type=str, default='faceformer', help='which model to train, faceformer, imitator_gen, imitator_stg01, imitator_stg02')
    parser.add_argument("--base_model_path", type=str, default='', help='path to previous round of training (for imitator_stg01 and 02')

    parser.add_argument("--batch_size", type=int, default=1, help='batch size')

    parser.add_argument("--delta", type=float, default=1e-5, help='dp delta')
    parser.add_argument("--noise_multiplier", type=float, default=1.0, help='dp noise_multiplier')
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help='dp max grad norm')
    args = parser.parse_args()

    if args.train_idx != -1:
        args.train_subjects = args.train_subjects.split(" ")[args.train_idx]


    if args.data_split=='horizontal':
        print('HORIZONTAL DATA SPLIT, setting vaild and test subjects to train subjects')
        print('data will be split 60/20/20 within speakers')
        # args.train_subjects = ' '.join([args.train_subjects, args.valid_subjects, args.test_subjects])
        args.valid_subjects=args.train_subjects
        args.test_subjects=args.train_subjects

    
    out_dir = f'{args.dir}/{args.dataset}/{args.save_path}/c_{args.num_clients}_e_{args.max_epoch}_aggr_{args.aggr}_ds_{args.data_split}_dp_{args.dp}_delta{args.delta}_nm_{args.noise_multiplier}_gn_{args.max_grad_norm}_tidx_{args.train_idx}_model_{args.model}'
    os.makedirs(out_dir, exist_ok=True)

    train_subjects_list = [i for i in args.train_subjects.split(" ")]
    train_subjects_list = get_train_subjects_partition(train_subjects_list, args.num_clients)

    from data_loader import get_dataloaders

    if args.dp=='opacus':
        # from imitator.models.opacus_nn_model_jp import imitator
        print('warning, importing default model with opacus flag!')
        from imitator.models.nn_model_jp import imitator
        # print('warning importing opacus 2 model')
        # from imitator.models.opacus_nn_model_jp_2 import imitator

    else:
        from imitator.models.nn_model_jp import imitator
    # from imitator.models.nn_model_jp import imitator



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
        def __init__(self, net, trainloader, valloader, optimizer, criterion, accountant, fileid_to_filename, cid):
            loss_cfg = {'full_rec_loss': 1.0, 'velocity_weight': 10.0}
            self.net = net
            self.trainloader = trainloader
            self.valloader = valloader
            self.optimizer = optimizer
            self.criterion = criterion
            self.cid = cid
            self.accountant = accountant
            self.train_metrics_path = f'{out_dir}/client_{cid}_train_results.csv'
            self.valid_metrics_path = f'{out_dir}/client_{cid}_valid_results.csv'
            self.custom_loss = Custom_errors(args.vertice_dim, loss_creterion=self.criterion, loss_dict=loss_cfg)
            self.fileid_to_filename = fileid_to_filename

        def get_parameters(self, config):
            return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

        def fit(self, parameters, config):
            print(f"training client {self.cid}")
            set_parameters(self.net, parameters)
            epoch_losses = train(args, self.trainloader, self.net, self.optimizer, self.criterion, self.custom_loss, self.accountant, self.fileid_to_filename, epoch=args.max_epoch)
            # evaluate after train epochs concluded
            metrics_dict = {'loss': epoch_losses}
            add_to_csv(self.train_metrics_path, metrics_dict)

            print(f"evaluating client {self.cid}")
            loss = eval(args, self.valloader, self.net, self.criterion, self.custom_loss, self.fileid_to_filename)
            # self.net = self.net.to_standard_module()
            accuracy = test(args, self.net, self.valloader, self.fileid_to_filename)
            # loss = eval(args, VALLOADER, self.net, self.criterion)
            # accuracy = test(args, self.net, VALLOADER)
            metrics_dict = {'loss': [loss], 'accuracy': [float(accuracy)]}
            if args.dp=='opacus':
                epsilon = self.accountant.get_epsilon(delta=args.delta)
                metrics_dict['epsilon'] = epsilon
            add_to_csv(self.valid_metrics_path, metrics_dict)
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
        train_subjects_subset = train_subjects_list[int(cid)] #the train subjects for a given client

        if args.model=='faceformer':
            from faceformer import Faceformer
            model = Faceformer(args)
        elif args.model.startswith('imitator'):
            #might have to do some funky stuff with the args first
            # args.num_identity_classes = 8 #maybe change this to all_train_subjects like in their code
            args.num_identity_classes = len(args.all_train_subjects.split()) #always all subjects
            args.num_dec_layers = 5
            args.fixed_channel = True
            args.style_concat = False
            model = imitator(args)
            if args.model=='imitator_stg01':
                print(f'training imitator_stg01, loading existing model at {args.base_model_path}')
                pretrained_model = torch.load(args.base_model_path)
                parameters = []
                for name, val in pretrained_model.items():
                    if name=='obj_vector.weight':
                        val[:, 0] = val[:, int(args.condition_idx)] #Set 0th speaker as condition idx (best one from paper)                        
                    parameters.append(val.cpu().numpy())
                set_parameters(model, parameters)
                print('freezing params for stg01')
                model = freeze_p1(model)
            elif args.model=='imitator_stg02':
                print(f'training imitator_stg02, loading existing model at {args.base_model_path}')
                pretrained_model = torch.load(args.base_model_path)
                parameters = [val.cpu().numpy() for _, val in pretrained_model.items()]
                set_parameters(model, parameters)
                print('doing stg02, freezing weights')
                model = freeze_p2(model)

        # to cuda
        assert torch.cuda.is_available()
        model = model.to(torch.device("cuda"))
        criterion = nn.MSELoss()
        # if args.dp=='opacus':
        #     #do NOT filter untrainable params bc it breaks opacus. Opacus SHOULD ignore these.
        #     # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        #     optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=args.lr)
        #     # for pid, p in enumerate(model.parameters()):
        #     #     try:
        #     #         print(pid, p.requires_grad)
        #     #     except Exception as e:
        #     #         print(e)
        #     #         continue
        #     # breakpoint()
        # else:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=args.lr)

        # dataset = get_dataloaders(args, return_test=False)

        dataset, fileid_to_filename = get_dataloaders(args, train_subjects_subset, splits=['train','valid'])
        
        trainloader = dataset['train']
        valloader = dataset['valid']
        
        accountant = None
        if args.dp=='opacus':
            # OPTION 1: Fully wrap the model etc in the standard way 
            # from opacus.grad_sample import GradSampleModule, GradSampleModuleExpandedWeights
            # from opacus import PrivacyEngine
            # # from register_grad_samplers import compute_td_grad_sample
            # privacy_engine = PrivacyEngine(secure_mode=False) #make secure for final training!
            # model = GradSampleModule(model)
            # model, optimizer, trainloader = privacy_engine.make_private(
            #     module=model,
            #     optimizer=optimizer,
            #     data_loader=trainloader,
            #     noise_multiplier=1.0,
            #     max_grad_norm=1.0,
            #     poisson_sampling=False
            #     # grad_sample_mode="ew"
            #     # grad_sample_mode="functorch"
            # )
            # accountant = privacy_engine.accountant

            # OPTION 2: Just wrap the optimizer
            # see: https://opacus.ai/tutorials/intro_to_advanced_features
            from opacus.optimizers import DPOptimizer
            sample_rate = 1 / len(trainloader)
            expected_batch_size = int(len(trainloader.dataset) * sample_rate)
            expected_batch_size = 1
            optimizer = DPOptimizer(
                optimizer=optimizer,
                noise_multiplier=args.noise_multiplier,
                max_grad_norm=args.max_grad_norm,
                expected_batch_size=expected_batch_size,
            )
            from opacus.accountants import RDPAccountant
            accountant = RDPAccountant()
            optimizer.attach_step_hook(accountant.get_optimizer_hook_fn(sample_rate=sample_rate))


        # Create a  single Flower client representing a single organization
        client = FlowerClient(model, trainloader, valloader, optimizer, criterion, accountant, fileid_to_filename, cid).to_client()

        #eval intitial parameters - really janky implementation just for debugging
        # npclient = client.numpy_client
        # if not os.path.exists(npclient.metrics_path):
        #     print(f"evaluating client {npclient.cid} (initial params)")
        #     loss = eval(args, npclient.valloader, npclient.net, npclient.criterion)
        #     accuracy = test(args, npclient.net, npclient.valloader)
        #     metrics_dict = {'loss': [loss], 'accuracy': [float(accuracy)]}
        #     add_to_csv(npclient.metrics_path, metrics_dict)

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
        if args.model=='faceformer':
            model = Faceformer(args)
        elif args.model.startswith('imitator'):
            args.num_identity_classes = len(args.all_train_subjects.split()) #always all subjects
            args.num_dec_layers = 5
            args.fixed_channel = True
            args.style_concat = False
            model = imitator(args)
        model = model.to(torch.device("cuda"))
        set_parameters(model, parameters)  # Update model with the latest parameters
        
        # dataset = get_dataloaders(args, return_test=False) #load every time for memory
        dataset, fileid_to_filename = get_dataloaders(args, train_subjects_subset=None, splits=['valid'])
        valloader = dataset['valid']

        # valloader = VALLOADER #use global valloader

        criterion = nn.MSELoss()
        loss_cfg = {'full_rec_loss': 1.0, 'velocity_weight': 10.0}        
        custom_loss = Custom_errors(args.vertice_dim, loss_creterion=criterion, loss_dict=loss_cfg)

        loss = eval(args, valloader, model, criterion, custom_loss, fileid_to_filename)
        accuracy = test(args, model, valloader, fileid_to_filename)
        
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
                if args.model=='faceformer':
                    net = Faceformer(args)
                elif args.model.startswith('imitator'):
                    args.num_identity_classes = len(args.all_train_subjects.split()) #always all subjects
                    args.num_dec_layers = 5
                    args.fixed_channel = True
                    args.style_concat = False
                    net = imitator(args)
                    
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
                    print('DELETING OLD MODEL')
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
    

    if args.dp=='fixed':
        #says to use noise >=1.0 but no advice on clipping norm. The paper seemed to go up to 20. Bad results with everything so far
        strategy = fl.server.strategy.DPFedAvgFixed(strategy, noise_multiplier=1.0, clip_norm=20, num_sampled_clients=args.num_clients, server_side_noising=True) 
    elif args.dp=='adaptive':
        #the adaptive one seems to start with much nicer defaults..!
        strategy = fl.server.strategy.DPFedAvgAdaptive(strategy, noise_multiplier=1.0, num_sampled_clients=args.num_clients, server_side_noising=True) 
        # strategy = fl.server.strategy.DifferentialPrivacyServerSideFixedClipping(strategy, noise_multiplier=1.0, clipping_norm=0.1, num_sampled_clients=args.num_clients)

        
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
            if os.path.exists(f"{out_dir}/model_round_{i-1}.pth"):
                os.remove(f"{out_dir}/model_round_{i-1}.pth") #delete old model to save space


    else:
        # Start simulation
        fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=args.num_clients,
            config=fl.server.ServerConfig(num_rounds=args.max_rounds),
            strategy=strategy,
            client_resources=client_resources,
        )