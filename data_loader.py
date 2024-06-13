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
    def __init__(self, data,subjects_dict, filename_to_fileid, fileid_to_filename, data_type="train",n_train_speakers=8):
        self.data = data
        self.len = len(self.data)
        self.subjects_dict = subjects_dict
        self.data_type = data_type
        # if self.data_type == "train":
        #     n_train_speakers = len(subjects_dict["train"])
        self.one_hot_labels = np.eye(n_train_speakers)
        self.filename_to_fileid = filename_to_fileid
        self.fileid_to_filename = fileid_to_filename

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        # seq_len, fea_dim
        file_name = self.data[index]["name"]
        fileid = self.filename_to_fileid[file_name[:-4]]
        audio = self.data[index]["audio"]
        vertice = self.data[index]["vertice"]
        template = self.data[index]["template"]
        if self.data_type == "train":
            subject = "_".join(file_name.split("_")[:-1])
            one_hot = self.one_hot_labels[self.subjects_dict["train"].index(subject)]
        else:
            one_hot = self.one_hot_labels
        return torch.FloatTensor(audio),torch.FloatTensor(vertice), torch.FloatTensor(template), torch.FloatTensor(one_hot), torch.LongTensor([fileid])

    def __len__(self):
        return self.len

def read_data(args, subjects, split):
    ''' Reads the data for the given subjects. 
        split determines what the full subject list is (where len(subjects) < len(split_subjects) eg for the federated learning setting with devices > 1)  
    '''
    print(f"Loading data split {split}...")
    data = defaultdict(dict)
    dataset_data = []
    filename_to_fileid = {}
    fileid_to_filename = {}
    all_subjects = eval(f'args.{split}_subjects') #train valid or test. 
    if subjects==None:
        subjects=all_subjects #for test and valid, always use all subjects
    if args.dir:
        dataset_path = os.path.join(args.dir, args.dataset)
    audio_path = os.path.join(dataset_path, args.wav_path)
    vertices_path = os.path.join(dataset_path, args.vertices_path)
    processor = Wav2Vec2Processor.from_pretrained(args.wav2vec_path)
    template_file = os.path.join(dataset_path, args.template_file)
    # with open(template_file, 'rb') as fin:
    try:
        templates = pickle.load(open(template_file, 'rb'))
    except UnicodeDecodeError:
        templates = pickle.load(open(template_file, 'rb'), encoding='latin1')

    
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
                    elif args.dataset == "hdtf":
                        data[key]["vertice"] = np.load(vertice_path,allow_pickle=True)[::2,:].reshape(-1, int(args.vertice_dim))
                    elif args.dataset == "BIWI":
                        data[key]["vertice"] = np.load(vertice_path,allow_pickle=True)
    subjects_dict = {}
    subjects_dict[split] = [i for i in all_subjects.split(" ")]

    if args.data_split=='horizontal':
        # vocaset_idx = np.arange(1,41)
        # n_train = int(len(vocaset_idx)*0.6) #60%
        # n_valid = int(len(vocaset_idx)*0.8) #20%
        # n_test = int(len(vocaset_idx)*1) #20%
        # assert n_train+n_valid+n_test==len(vocaset_idx)
        # train_idx = vocaset_idx[:n_train]
        # valid_idx = vocaset_idx[n_train:n_valid]
        # test_idx = vocaset_idx[n_valid:n_test]
        # splits = {'vocaset':{'train':train_idx,'valid':valid_idx,'test':test_idx}, 'BIWI':{'train':range(1,33),'valid':range(33,37),'test':range(37,41)}}
        splits = {'vocaset':{'train':range(1,25),'valid':range(25,33),'test':range(33,41)}, 'BIWI':{'train':range(1,33),'valid':range(33,37),'test':range(37,41)}, 'hdtf':{'train':range(1,8),'valid':range(8,10),'test':range(10,13)}}
    elif args.data_split=='vertical':
        if args.dataset=='vocaset':
            # print('WARNING: DEFAULT SPLIT ONLY USES HALF THE TEST AND VALID SETS!') #used to be (21,41) - fixed
            splits = {'vocaset':{'train':range(1,41),'valid':range(1,41),'test':range(1,41)}, 
                  'BIWI':{'train':range(1,33),'valid':range(33,37),'test':range(37,41)}, 
                  'hdtf': {'train':range(1,13),'valid':range(1,13),'test':range(1,13)}}
    elif args.data_split=='stg':
        #imitator style adaptation, just take a few utts (taken from config)
        splits = {'vocaset':{'train':range(1,5),'valid':range(19,21),'test':range(21,41)}}

    else:
        raise Exception('args.data_split unknown')
    for i, (k, v) in enumerate(data.items()):
        if split=='test':
            i+=100000
        elif split=='valid':
            i-=100000
        filename_to_fileid[k[:-4]] = i
        fileid_to_filename[i] = k[:-4]
        subject_id = "_".join(k.split("_")[:-1])
        sentence_id = int(k.split(".")[0][-2:])
        if subject_id in subjects.split() and sentence_id in splits[args.dataset][split]:
            # dataset_data.append(v)
            dataset_data.append(v) #append only the fileid, and return the mapping as a dict (opacus cant handle strings in the collate fn)
    print(f'loaded data {split}: length {len(dataset_data)}')
    return dataset_data, subjects_dict, fileid_to_filename, filename_to_fileid

def pad_collate_fn(batch):
    #allow for batches larger than 1
    # batch is a list of tuples: 
    # [(audio, vertice, template, one_hot, file_name), ..]
    #get the max len
    audio_max_len = max([len(x[0]) for x in batch])
    vertice_max_len = max([len(x[1]) for x in batch])
    padded_audios = []
    padded_vertices = []
    for i in batch:
        #pad audio and vertice (just repeat pad final index)
        audio = i[0]
        audio_pad = audio_max_len-len(audio)
        padded_audio = torch.nn.functional.pad(i[0], (0,audio_pad), 'constant', audio[-1]) #repeat final val
        padded_audios.append(padded_audio)

        #there must be a real function for this.. but this was the best i could do 
        vertice = i[1]
        padded_vertice = torch.zeros(vertice_max_len, vertice.shape[1])
        padded_vertice[:len(vertice), :] = vertice
        padded_vertice[len(vertice):, :] = vertice[-1]
        padded_vertices.append(padded_vertice)
    #return as list of tensors
    padded_batch = (
        torch.stack(padded_audios), 
        torch.stack(padded_vertices), 
        torch.stack([x[2] for x in batch]),
        torch.stack([x[3] for x in batch]),
        # [x[4] for x in batch], #these are strings
        torch.stack([x[4] for x in batch]), #jp edit, made them IDs
    )
    return padded_batch

def get_dataloaders(args, train_subjects_subset=None, splits=['train','valid','test']):
    if args.model.startswith('imitator'):
        n_train_speakers = len(args.all_train_subjects.split())
        if train_subjects_subset==None:
            train_subjects_subset = args.all_train_subjects

    else:
        n_train_speakers = len(args.train_subjects.split())
        if train_subjects_subset==None:
            train_subjects_subset = args.train_subjects
    dataset = {}
    fileid_to_filename = {}
    for split in splits:
        if split=='train':
            subjects = train_subjects_subset
            batch_size=args.batch_size
        else:
            subjects = None
            batch_size=1

        # subjects = train_subjects_subset #filters for test and valid also - desirable?
        dataset_data, subjects_dict, fid_to_fn, filename_to_fileid = read_data(args, subjects, split=split)
        fileid_to_filename = fileid_to_filename | fid_to_fn 

        dataset_data = Dataset(dataset_data, subjects_dict, filename_to_fileid, fid_to_fn, split, n_train_speakers)
        shuffle = False if split=='train' else False
        # dataset[split] = data.DataLoader(dataset=dataset_data, shuffle=shuffle)
        dataset[split] = data.DataLoader(dataset=dataset_data, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_collate_fn)
    return dataset


if __name__ == "__main__":
    get_dataloaders()
    