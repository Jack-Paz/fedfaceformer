import os, datetime, glob, importlib
# from omegaconf import OmegaConf
import numpy as np
import json
# from pytorch_lightning import seed_everything
import collections, functools, operator
import torch
import torch.nn as nn
torch.backends.cudnn.deterministic = True
from timeit import default_timer as timer
from imitator.utils.init_from_config import instantiate_from_config
from argparse import ArgumentParser

from main import count_parameters

def process_result_dict(results_dict):
    # combine the results dicts
    loss_dicts = [batch['results'] for batch in results_dict]
    # add
    combined = dict(functools.reduce(operator.add, map(collections.Counter, loss_dicts)))
    # average the loss
    average_loss = {key: combined[key] / len(loss_dicts) for key in combined.keys()}
    return average_loss

class test_dataset_wise():
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args

        ### create the one-hot labels
        one_hot_labels = np.eye(8)
        self.one_hot_labels = torch.from_numpy(one_hot_labels).view(1, 8, 8).float()

        ### create the losses
        from imitator.utils.losses import Custom_errors
        from FLAMEModel.flame_masks import get_flame_mask
        loss_cfg = {}
        loss = nn.MSELoss()
        mask = get_flame_mask()
        self.lips_idxs = mask.lips
        lip_mask = torch.zeros((1, 5023, 3))
        lip_mask[0, self.lips_idxs] = 1.0
        self.lip_mask = lip_mask.view(1, -1)
        self.custom_loss = Custom_errors(15069, loss_creterion=loss, loss_dict=loss_cfg)

        ### create a render
        if self.args.render_results:
            from imitator.utils.render_helper import render_helper
            self.rh = render_helper()

    def run_loop_with_condition_test(self, dataloader, model, condition_id, eval_static_mesh=False):
        # if len(model.nn_model.train_subjects) > 1:
        #     condition_subject = model.nn_model.train_subjects[condition_id]
        # else:
        #     ### personalized style model
        #     condition_subject = model.nn_model.train_subjects[0]
        # print("Current condition subject", condition_subject)

        results_dict_list = []

        full_reconstruction_mm = []
        lip_reconstruction_mm = []
        lip_max_l2 = []

        # set the modelt to eval for computation
        print("Set the model for the evaluation")
        model = model.eval()
        if condition_id==-1:
            print('testing all conditions and picking best')
            condition_id_list=np.arange(8)
        else:
            condition_id_list = [condition_id]
        print("Total sequence to run in the dataloader", len(dataloader))
        for batch in dataloader:
            audio, vertice, template, one_hot_all, file_name = batch
            print("file_name", file_name)
            audio = audio.to(torch.device("cuda"))
            vertice = vertice.to(torch.device("cuda"))
            template = template.to(torch.device("cuda"))
            frmms, lrmms, lipmaxs = [], [], []
            for cid in condition_id_list:
                one_hot = self.one_hot_labels[:, cid, :].to(torch.device("cuda"))
                if eval_static_mesh:
                    print('NOTICE: EVALUATING STATIC MESH')
                    prediction = template.repeat(1, vertice.shape[1], 1)
                else:
                    prediction = model.predict(audio, template, one_hot)
                pred_len = prediction.shape[1]
                vertice = vertice[:, :pred_len]
                frmms.append(self.custom_loss.error_in_mm(prediction, vertice).cpu().numpy())
                lrmms.append(self.custom_loss.compute_masked_error_in_mm(prediction, vertice, self.lip_mask.to(torch.device('cuda'))).cpu().numpy())
                lipmaxs.append(self.custom_loss.lip_max_l2(prediction, vertice, self.lip_mask.to(torch.device('cuda'))).item())
            best_idx = np.argmin(lipmaxs) #choose best lipmax 
            frmm, lrmm, lipmax = frmms[best_idx], lrmms[best_idx], lipmaxs[best_idx]
            # reconstruction in mm
            full_reconstruction_mm.append(frmm)
            lip_reconstruction_mm.append(lrmm)
            # lip metrics
            lip_max_l2.append(lipmax)

        # simple metric rec loss
        out_dict = {
            "full_reconstruction_mm": np.mean(full_reconstruction_mm),
            "lip_reconstruction_mm": np.mean(lip_reconstruction_mm),
            "lip_max_l2": np.mean(lip_max_l2),
                    }

        results_dict_list.append({'results': out_dict,
                                    'predict': prediction.detach().cpu(),
                                    'seq':file_name}
                                    )

        return results_dict_list

    def run_test(self, model, data, logdir, dataset_to_eval, condition=2):
        out_dir = os.path.join(logdir, f"voca_eval_with_fixed_test_cond_{condition}")
        os.makedirs(out_dir, exist_ok=True)

        test_data = data

        results_dict = self.run_loop_with_condition_test(test_data, model, condition)

        ### process and dump the metrics
        metrics = process_result_dict(results_dict)
        with open(os.path.join(out_dir, 'results.json'), 'w') as file:
            file.write(json.dumps(metrics, indent=4))

        for seq in results_dict:
            pred = seq["predict"]
            file_name = seq["seq"]
            seq_name = file_name[0].replace(".wav", "")
            if self.args.render_results:
                vid_dir = os.path.join(out_dir, "vid")
                os.makedirs(vid_dir, exist_ok=True)
                if os.getenv("VOCASET_PATH"):
                    audio_file = os.path.join(os.getenv("VOCASET_PATH"),
                                          data.data_cfg["wav_path"],
                                          file_name[0])
                else:
                    audio_file = os.path.join(os.getenv("HOME"),
                                            data.data_cfg["dataset_root"],
                                            data.data_cfg["wav_path"],
                                            file_name[0])
                self.rh.visualize_meshes(vid_dir, seq_name, pred.reshape(-1, 5023,3), audio_file)

            if self.args.dump_results:
                dump_dir = os.path.join(out_dir, "dump")
                os.makedirs(dump_dir, exist_ok=True)
                out_file = os.path.join(dump_dir, seq_name+".npy")
                np.save(out_file, pred.reshape(-1, 5023, 3).numpy())
                print("Dumping file", out_file)



if __name__ == "__main__":

    start = timer()

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
    parser.add_argument("--max_epoch", type=int, default=100, help='number of epochs')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--template_file", type=str, default="templates.pkl", help='path of the personalized templates')
    parser.add_argument("--save_path", type=str, default="save", help='path of the trained models')
    parser.add_argument("--result_path", type=str, default="result", help='path to the predictions')
    parser.add_argument("--train_subjects", type=str, default="FaceTalk_170728_03272_TA"
       " FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA"
       " FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA"
       " FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA")
    parser.add_argument("--val_subjects", type=str, default="FaceTalk_170811_03275_TA"
       " FaceTalk_170908_03277_TA")
    parser.add_argument("--test_subjects", type=str, default="FaceTalk_170809_00138_TA"
       " FaceTalk_170731_00024_TA")
    parser.add_argument('--model_path', type=str, default='vocaset/save_backup/baseline/100_model.pth')
    parser.add_argument('--condition_id', type=str, default='-1', help='condition id OR -1 to test on all ids and choose the min')
    parser.add_argument('-d', '--dump_results', default=False, action='store_true')
    parser.add_argument('-r', '--render_results', default=False, action='store_true')
    parser.add_argument("--train_idx", type=int, default=-1, help='index of speaker to train on for individual run, -1 = train on all speakers')
    parser.add_argument('--test_random_initialisation', action='store_true', default=False)

    args = parser.parse_args()
    from faceformer import Faceformer
    from data_loader import get_dataloaders

    #build model
    train_subjects_list = args.train_subjects.split()
    if args.train_idx!=-1:
        args.train_subjects = train_subjects_list[args.train_idx]

    #build model
    model = Faceformer(args)

    #load model
    if args.test_random_initialisation:
        print('WARNING: testing random initialisation! results will be measured on a RANDOM model')
    else:
        print(f'loading model: {args.model_path}')
        state_dict = torch.load(args.model_path)
        model.load_state_dict(state_dict, strict=True)
        print("model parameters: ", count_parameters(model))

    
    assert torch.cuda.is_available()
    model.to(torch.device('cuda'))
    dataset = get_dataloaders(args)
    #save the test outputs to result/ folder
    data = dataset['test']

    tester = test_dataset_wise(args=args)
    logdir = os.path.dirname(args.model_path)
    tester.run_test(model, data, logdir, None, condition=int(args.condition_id))
    # tester = test_dataset_wise(args=opt)
    # print()

    # tester.run_test(model, data, logdir,
    #                           opt.data_to_eval, condition=int(data_cfg.conditiion_id))

    end = timer()
    print("\n\nTime to take to run the tesing suite in sec", end - start)