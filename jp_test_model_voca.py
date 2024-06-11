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
import matplotlib.pyplot as plt

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
        if args.aggr=='avg':
            one_hot_labels = np.zeros(len(args.all_train_subjects.split()))
            self.one_hot_labels = torch.FloatTensor(one_hot_labels).unsqueeze(0).unsqueeze(-1)
            # self.one_hot_labels = torch.FloatTensor(self.one_hot_labels)
        else:
            one_hot_labels = np.eye(len(args.all_train_subjects.split()))
            self.one_hot_labels = torch.from_numpy(one_hot_labels).unsqueeze(0).float()
        # self.one_hot_labels = torch.from_numpy(one_hot_labels).view(1, 8, 8).float()
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

    def run_loop_with_condition_test(self, dataloader, model, condition_id, eval_static_mesh=False, tsne=None, aggr='avg'):
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
            # breakpoint()
            audio, vertice, template, one_hot_all, fileid = batch
            file_name = dataloader.dataset.fileid_to_filename[int(fileid)]
            audio = audio.to(torch.device("cuda"))
            vertice = vertice.to(torch.device("cuda"))
            template = template.to(torch.device("cuda"))
            frmms, lrmms, lipmaxs = [], [], []
            predictions = []
            for cid in condition_id_list:
                cid = int(cid)
                one_hot = self.one_hot_labels[:, cid, :].to(torch.device("cuda"))
                if eval_static_mesh:
                    print('NOTICE: EVALUATING STATIC MESH')
                    prediction = template.repeat(1, vertice.shape[1], 1)
                elif tsne is not None:
                    if aggr=='avg':
                        #1-d input layer?
                        one_hot = torch.FloatTensor([[cid]]).to(torch.device('cuda'))
                    # one_hot = one_hot_all
                    prediction = model.return_hidden_state(audio, template, one_hot, layer_idx=tsne)
                    predictions.append(prediction.detach().cpu())
                    continue #dont do the rest of the loop
                else:
                    prediction = model.predict(audio, template, one_hot)
                pred_len = prediction.shape[1]
                vertice = vertice[:, :pred_len]
                frmms.append(self.custom_loss.error_in_mm(prediction, vertice).cpu().numpy())
                lrmms.append(self.custom_loss.compute_masked_error_in_mm(prediction, vertice, self.lip_mask.to(torch.device('cuda'))).cpu().numpy())
                lipmaxs.append(self.custom_loss.lip_max_l2(prediction, vertice, self.lip_mask.to(torch.device('cuda'))).item())
            if tsne is not None:
                results_dict_list.append({'predict': torch.stack(predictions), 'seq': file_name})
                continue
            best_idx = np.argmin(lipmaxs) #choose best lipmax 
            frmm, lrmm, lipmax = frmms[best_idx], lrmms[best_idx], lipmaxs[best_idx]
            # reconstruction in mm
            full_reconstruction_mm.append(frmm)
            lip_reconstruction_mm.append(lrmm)
            # lip metrics
            lip_max_l2.append(lipmax)

        if tsne:
            return results_dict_list

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
        model_name = os.path.basename(args.model_path)[:-4]
        out_dir = os.path.join(logdir, f"eval_model_{model_name}_train_idx_{args.train_idx}_cond_{condition}")
        os.makedirs(out_dir, exist_ok=True)

        test_data = data
        results_dict = self.run_loop_with_condition_test(test_data, model, condition,tsne=args.tsne, aggr=args.aggr)

        if args.tsne is not None:
            spk_to_spkid = {
                'FaceTalk_170728_03272': 0,
                'FaceTalk_170904_00128': 1,
                'FaceTalk_170725_00137': 2,
                'FaceTalk_170915_00223': 3,
                'FaceTalk_170811_03274': 4,
                'FaceTalk_170913_03279': 5, 
                'FaceTalk_170904_03276': 6, 
                'FaceTalk_170912_03278': 7,
                'FaceTalk_170809_00138': 8, 
                'FaceTalk_170731_00024': 9
                }
            results_list = results_dict #not a dict if tsne
            from sklearn.manifold import TSNE
            all_data, test_ids, condition_ids = [], [], []
            if args.condition_id==-1:
                condition_idxs = range(len(train_subjects_list))
            else:
                condition_idxs = [int(args.condition_id)]
            # condition_idxs = 
            for condition_idx in condition_idxs:
                for example in results_list:
                    n_speakers, bsz, seq_len, hidden_dim = example['predict'].shape
                    # condition_idx = condition_idx  # Select a speaker
                    speaker_data = example['predict'][condition_idx].squeeze(0)  # This removes the batch size dimension
                    speaker = '_'.join(example['seq'].split('_')[:-2])
                    
                    spkid = spk_to_spkid[speaker]
                    test_ids.append(spkid)
                    condition_ids.append(condition_idx)
                    # setting_labels.append(f'spk_{spkid}_cond_{condition_idx}')
                    speaker_data = np.asarray(speaker_data)
                    average_data = np.mean(speaker_data, axis=0).reshape(1, -1) #might have to average bc seq_len is different
                    all_data.append(average_data)
            all_data = np.array(all_data).squeeze(1)
            test_ids = np.array(test_ids)
            condition_ids = np.array(condition_ids)
            tsne = TSNE(perplexity=int(len(all_data)/2), n_components=2, random_state=42)  # n_components=2 for 2D plot

            tsne_results = tsne.fit_transform(all_data)
            color_map = plt.cm.get_cmap('tab10', 10)
            markers = ['+','o','x','1','H','*','D','>', '<', '|']
            for i in range(len(tsne_results)):
                color = color_map(condition_ids[i])

                marker = markers[test_ids[i]]
                # marker = '+' if test_ids[i] == 0 else 'o'
                plt.scatter(tsne_results[i, 0], tsne_results[i, 1], color=color, marker=marker, alpha=0.6)

            # Create a custom legend for condition speakers
            for i in range(len(set(condition_ids))):
                plt.scatter([], [], color=color_map(i), label=f'Condition {i+1}')
            for i in range(len(set(test_ids))):
                # Add legend entries for test speakers
                plt.scatter([], [], color='black', marker=markers[i], label=f'Test ID {i}')
                # plt.scatter([], [], color='black', marker='o', label='Test ID 2')

            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.tight_layout()
            plt.title('t-SNE of Model Representations')
            plt.xlabel('t-SNE Dimension 1')
            plt.ylabel('t-SNE Dimension 2')
            # plt.show()

            # print('done getting tsne')
            # plt.figure(figsize=(12, 6))
            # plt.subplot(1, 2, 2)
            # scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=setting_ids, cmap=color_map, alpha=0.6)
            # plt.colorbar(scatter, ticks=np.unique(setting_ids))
            # plt.title('t-SNE of Averaged Data')
            plt.savefig(f'{out_dir}/tsne.png')

            # do tsne and return
            return   

        ### process and dump the metrics
        metrics = process_result_dict(results_dict)
        print(f'writing results to {out_dir}/results.json')
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
    parser.add_argument("--dir", type=str, default='.', help='path to working dir')

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
    parser.add_argument("--all_train_subjects", type=str, default="FaceTalk_170728_03272_TA FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA")
    parser.add_argument("--valid_subjects", type=str, default="FaceTalk_170811_03275_TA"
       " FaceTalk_170908_03277_TA")
    parser.add_argument("--test_subjects", type=str, default="FaceTalk_170809_00138_TA"
       " FaceTalk_170731_00024_TA")
    parser.add_argument('--model_path', type=str, default='vocaset/save_backup/baseline/100_model.pth')
    parser.add_argument('--condition_id', type=str, default='-1', help='condition id OR -1 to test on all ids and choose the min')
    parser.add_argument('-d', '--dump_results', default=False, action='store_true')
    parser.add_argument('-r', '--render_results', default=False, action='store_true')
    parser.add_argument("--train_idx", type=int, default=-1, help='index of speaker to train on for individual run, -1 = train on all speakers')
    parser.add_argument('--test_random_initialisation', action='store_true', default=False)
    parser.add_argument("--aggr", type=str, default="avg", help='avg | mask - which aggregation method to use')
    parser.add_argument("--wav2vec_path", type=str, default="/home/paz/data/wav2vec2-base-960h", help='wav2vec path for the faceformer model')
    parser.add_argument("--data_split", type=str, default="vertical", help='vertical | horziontal - vertical=split on speakers, horzontal=split some of each train speaker for test and valid')
    parser.add_argument("--model", type=str, default='faceformer', help='which model to train, faceformer or imitator')
    parser.add_argument("--batch_size", type=int, default=1, help='batch size (1 for test?)')

    parser.add_argument("--tsne", type=int, nargs='?', default=None, help='save t_sne to model folder, wont run normal testing, int=layer idx')

    parser.add_argument("--num_identity_classes", type=int, default=8, help='n speakers / size of initial layer')

    parser.add_argument("--dp", type=str, default='none')

    parser.add_argument("--test_on_ood", action='store_true', help='also test on the 2 OOD speakers (horizontal only)')

    args = parser.parse_args()
    from faceformer import Faceformer
    from data_loader import get_dataloaders

    if args.data_split=='horizontal':
        print('HORIZONTAL DATA SPLIT, setting vaild and test subjects to train subjects')
        print('data will be split 60/20/20 within speakers')
        # args.train_subjects = ' '.join([args.train_subjects, args.valid_subjects, args.test_subjects])
        args.valid_subjects=args.train_subjects
        args.test_subjects=args.train_subjects

    #build model
    train_subjects_list = args.train_subjects.split()
    if args.aggr=='avg':
        if args.train_idx!=-1:
            args.train_subjects = train_subjects_list[args.train_idx]

    #build model
    if args.model=='faceformer':
        from faceformer import Faceformer
        model = Faceformer(args)
    elif args.model=='imitator':
        print('loading imitator')
        from imitator.models.nn_model_jp import imitator
        #might have to do some funky stuff with the args first
        args.num_identity_classes = len(args.all_train_subjects.split())
        args.num_dec_layers = 5
        args.fixed_channel = True
        args.style_concat = False
        model = imitator(args)

    #load model
    if args.test_random_initialisation:
        print('WARNING: testing random initialisation! results will be measured on a RANDOM model')
    else:
        print(f'loading model: {args.model_path}')
        state_dict = torch.load(args.model_path)
        if 'state_dict' in state_dict.keys():
            state_dict = state_dict['state_dict'] #for pretrained models
            state_dict = {'.'.join(x.split('.')[1:]): y for x, y in state_dict.items()} #remove nn_model. from names
            model.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict, strict=True)
        print("model parameters: ", count_parameters(model))

    
    assert torch.cuda.is_available()
    model.to(torch.device('cuda'))

    # if args.aggr=='avg':
    #     train_subjects_subset = train_subjects_list[args.train_idx]
    # else:
    #     train_subjects_subset=None
    train_subjects_subset = train_subjects_list[args.train_idx]
    
    dataset = get_dataloaders(args, train_subjects_subset, splits=['test'])
    data = dataset['test']
    if args.data_split=='horizontal' and args.test_on_ood==True:
        #hacky: get two full test speakers as well
        args.data_split='vertical'
        args.test_subjects = "FaceTalk_170809_00138_TA FaceTalk_170731_00024_TA"
        dataset = get_dataloaders(args, train_subjects_subset, splits=['test'])
        test_data = dataset['test']
        args.data_split='horizontal'
        data.dataset.data.extend(test_data.dataset.data)
        data.dataset.len = len(data.dataset.data)
        data.dataset.filename_to_fileid = data.dataset.filename_to_fileid | test_data.dataset.filename_to_fileid
        data.dataset.fileid_to_filename = data.dataset.fileid_to_filename | test_data.dataset.fileid_to_filename 
        # print('WARNING: FILEID_TO_FILENAME PROBABLY BROKEN')
        #Fixed by adding an offset of 100k to train and valid fileids 
    
    #save the test outputs to result/ folder
    tester = test_dataset_wise(args=args)
    logdir = os.path.dirname(args.model_path)
    tester.run_test(model, data, logdir, None, condition=int(args.condition_id))

    end = timer()
    print("\n\nTime to take to run the tesing suite in sec", end - start)