import os
import torch
from collections import defaultdict
from torch.utils import data
import copy
import numpy as np
import pickle
from tqdm import tqdm
import random,math
from transformers import Wav2Vec2FeatureExtractor,Wav2Vec2Processor
import librosa    

class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data,subjects_dict,data_type="train",n_train_speakers=8):
        self.data = data
        self.len = len(self.data)
        self.subjects_dict = subjects_dict
        self.data_type = data_type
        # if self.data_type == "train":
        #     n_train_speakers = len(subjects_dict["train"])
        self.one_hot_labels = np.eye(n_train_speakers)

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        # seq_len, fea_dim
        file_name = self.data[index]["name"]
        audio = self.data[index]["audio"]
        vertice = self.data[index]["vertice"]
        template = self.data[index]["template"]
        if self.data_type == "train":
            subject = "_".join(file_name.split("_")[:-1])
            one_hot = self.one_hot_labels[self.subjects_dict["train"].index(subject)]
        else:
            one_hot = self.one_hot_labels
        return torch.FloatTensor(audio),torch.FloatTensor(vertice), torch.FloatTensor(template), torch.FloatTensor(one_hot), file_name

    def __len__(self):
        return self.len

# def read_data(args):
#     print("Loading data...")
#     data = defaultdict(dict)
#     train_data = []
#     valid_data = []
#     test_data = []

#     audio_path = os.path.join(args.dataset, args.wav_path)
#     vertices_path = os.path.join(args.dataset, args.vertices_path)
#     # processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
#     # processor = Wav2Vec2Processor.from_pretrained("/home/paz/data/wav2vec2-base-960h")
#     processor = Wav2Vec2Processor.from_pretrained(args.wav2vec_path)
#     template_file = os.path.join(args.dataset, args.template_file)
#     with open(template_file, 'rb') as fin:
#         templates = pickle.load(fin,encoding='latin1')
    
#     for r, ds, fs in os.walk(audio_path):
#         # fs_list = tqdm(fs)
#         fs_list = fs
#         for f in fs_list:
#             if f.endswith("wav"):
#                 wav_path = os.path.join(r,f)
#                 speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
#                 input_values = np.squeeze(processor(speech_array,sampling_rate=16000).input_values)
#                 key = f.replace("wav", "npy")
#                 data[key]["audio"] = input_values
#                 subject_id = "_".join(key.split("_")[:-1])
#                 temp = templates[subject_id]
#                 data[key]["name"] = f
#                 data[key]["template"] = temp.reshape((-1)) 
#                 vertice_path = os.path.join(vertices_path,f.replace("wav", "npy"))
#                 if not os.path.exists(vertice_path):
#                     del data[key]
#                 else:
#                     if args.dataset == "vocaset":
#                         data[key]["vertice"] = np.load(vertice_path,allow_pickle=True)[::2,:]#due to the memory limit
#                     elif args.dataset == "BIWI":
#                         data[key]["vertice"] = np.load(vertice_path,allow_pickle=True)

#     subjects_dict = {}
#     subjects_dict["train"] = [i for i in args.train_subjects.split(" ")]
#     subjects_dict["val"] = [i for i in args.val_subjects.split(" ")]
#     subjects_dict["test"] = [i for i in args.test_subjects.split(" ")]

#     splits = {'vocaset':{'train':range(1,41),'val':range(21,41),'test':range(21,41)},
#      'BIWI':{'train':range(1,33),'val':range(33,37),'test':range(37,41)}}
   
#     for k, v in data.items():
#         subject_id = "_".join(k.split("_")[:-1])
#         sentence_id = int(k.split(".")[0][-2:])
#         if subject_id in subjects_dict["train"] and sentence_id in splits[args.dataset]['train']:
#             train_data.append(v)
#         if subject_id in subjects_dict["val"] and sentence_id in splits[args.dataset]['val']:
#             valid_data.append(v)
#         if subject_id in subjects_dict["test"] and sentence_id in splits[args.dataset]['test']:
#             test_data.append(v)

#     print(len(train_data), len(valid_data), len(test_data))
#     return train_data, valid_data, test_data, subjects_dict

# def get_dataloaders(args, return_test=True):
#     dataset = {}
#     train_data, valid_data, test_data, subjects_dict = read_data(args)
#     train_data = Dataset(train_data,subjects_dict,"train")
#     dataset["train"] = data.DataLoader(dataset=train_data, batch_size=1, shuffle=True)
#     valid_data = Dataset(valid_data,subjects_dict,"val")
#     dataset["valid"] = data.DataLoader(dataset=valid_data, batch_size=1, shuffle=False)
#     if return_test:
#         test_data = Dataset(test_data,subjects_dict,"test")
#         dataset["test"] = data.DataLoader(dataset=test_data, batch_size=1, shuffle=False)
#     return dataset

def read_data(args, subjects, split):
    ''' Reads the data for the given subjects. 
        split determines what the full subject list is (where len(subjects) < len(split_subjects) eg for the federated learning setting with devices > 1)  
    '''
    print(f"Loading data split {split}...")
    data = defaultdict(dict)
    dataset_data = []
    all_subjects = eval(f'args.{split}_subjects') #train valid or test. 
    if subjects==None:
        subjects=all_subjects #for test and valid, always use all subjects
    audio_path = os.path.join(args.dataset, args.wav_path)
    vertices_path = os.path.join(args.dataset, args.vertices_path)
    processor = Wav2Vec2Processor.from_pretrained(args.wav2vec_path)
    template_file = os.path.join(args.dataset, args.template_file)
    with open(template_file, 'rb') as fin:
        templates = pickle.load(fin,encoding='latin1')
    
    for r, ds, fs in os.walk(audio_path):
        # fs_list = tqdm(fs)
        fs_list = fs
        for f in fs_list:
            subject = '_'.join(f.split('_')[:-1])
            if subject not in subjects.split():
                continue
            if f.endswith("wav"):
                wav_path = os.path.join(r,f)
                speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
                input_values = np.squeeze(processor(speech_array,sampling_rate=16000).input_values)
                key = f.replace("wav", "npy")
                data[key]["audio"] = input_values
                subject_id = "_".join(key.split("_")[:-1])
                temp = templates[subject_id]
                data[key]["name"] = f
                data[key]["template"] = temp.reshape((-1)) 
                vertice_path = os.path.join(vertices_path,f.replace("wav", "npy"))
                if not os.path.exists(vertice_path):
                    del data[key]
                else:
                    if args.dataset == "vocaset":
                        data[key]["vertice"] = np.load(vertice_path,allow_pickle=True)[::2,:]#due to the memory limit
                    elif args.dataset == "BIWI":
                        data[key]["vertice"] = np.load(vertice_path,allow_pickle=True)
    subjects_dict = {}
    subjects_dict[split] = [i for i in all_subjects.split(" ")]

    splits = {'vocaset':{'train':range(1,41),'valid':range(21,41),'test':range(21,41)},
     'BIWI':{'train':range(1,33),'valid':range(33,37),'test':range(37,41)}}
    for k, v in data.items():
        subject_id = "_".join(k.split("_")[:-1])
        sentence_id = int(k.split(".")[0][-2:])
        if subject_id in subjects.split() and sentence_id in splits[args.dataset][split]:
            dataset_data.append(v)

    print(f'loaded data {split}: length {len(dataset_data)}')
    return dataset_data, subjects_dict

def get_dataloaders(args, train_subjects_subset=None, splits=['train','valid','test']):
    n_train_speakers = len(args.train_subjects.split())
    if train_subjects_subset==None:
        train_subjects_subset = args.train_subjects
    dataset = {}
    for split in splits:
        if split=='train':
            subjects = train_subjects_subset
        else:
            subjects = None
        dataset_data, subjects_dict = read_data(args, subjects, split=split)
        dataset_data = Dataset(dataset_data, subjects_dict, split, n_train_speakers)
        shuffle = False if split=='train' else False
        dataset[split] = data.DataLoader(dataset=dataset_data, batch_size=1, shuffle=shuffle)
    return dataset


if __name__ == "__main__":
    get_dataloaders()
    