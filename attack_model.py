'''
PREDICT SPEAKER:
 - Take trained model outputs
   - Train on model outputs from training data..?
   - Test on model outputs on test data?

Membership Inference Attacks:
 - Train a classifier to detect whether a sample was in the train set or not
 - Just analyse the logits (in-samples will be more confident than out-samples)
 - Is this also true for speakers regardless of whether theyre in or out?


'''
import os
import argparse
parser = argparse.ArgumentParser()
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


parser.add_argument('--experiment', type=str, default='speaker_identification', help='which experiment to run: speaker_identification | membership_inference')

args = parser.parse_args()

from faceformer import Faceformer
from data_loader import get_dataloaders

from jp_test_model_voca import test_dataset_wise

import torch
import torch.nn as nn

from tqdm import tqdm

spk_to_spkid = {
      'FaceTalk_170728_03272': 0,
      'FaceTalk_170904_00128': 1,
      'FaceTalk_170725_00137': 2,
      'FaceTalk_170915_00223': 3,
      'FaceTalk_170811_03274': 4,
      'FaceTalk_170913_03279': 5, 
      'FaceTalk_170904_03276': 6, 
      'FaceTalk_170912_03278': 7,
      # 'FaceTalk_170809_00138': 8,
      # 'FaceTalk_170731_00024': 9
      }

class SpeakerClassifier(nn.Module):
    def __init__(self, input_size, nhu, n_speakers):
        super(SpeakerClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, nhu)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(nhu, n_speakers)

      #   self.fc_out = nn.Linear(input_size, n_speakers)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc_out(out)
        
      #   out = self.fc_out(x)
        return out


train_subjects_list = args.train_subjects.split()
def load_main_model(args):
   if args.aggr=='avg':
      if args.train_idx!=-1:
         args.train_subjects = train_subjects_list[args.train_idx]

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

   model = model.to(torch.device('cuda'))
   print(f'loading model: {args.model_path}')
   state_dict = torch.load(args.model_path)
   if 'state_dict' in state_dict.keys():
      state_dict = state_dict['state_dict'] #for pretrained models
      state_dict = {'.'.join(x.split('.')[1:]): y for x, y in state_dict.items()} #remove nn_model. from names
      model.load_state_dict(state_dict)
   else:
      model.load_state_dict(state_dict, strict=True)
   # print("model parameters: ", count_parameters(model))
   return model

def get_data(args, model):
   tester = test_dataset_wise(args=args)
   # logdir = os.path.dirname(args.model_path)
   # tester.run_test(model, data, logdir, None, condition=int(args.condition_id))
   args.train_subjects = ' '.join(train_subjects_list)
   train_subjects_subset = ' '.join(train_subjects_list)
   if args.data_split=='horizontal':
      print('HORIZONTAL DATA SPLIT, setting vaild and test subjects to train subjects')
      print('data will be split 60/20/20 within speakers')
      # args.train_subjects = ' '.join([args.train_subjects, args.valid_subjects, args.test_subjects])
      args.valid_subjects=args.train_subjects
      args.test_subjects=args.train_subjects

   dataset = get_dataloaders(args, train_subjects_subset, splits=['train', 'test'])

   test_data = dataset['test']
   train_data = dataset['train']

   condition=args.condition_id
   train_results_list = tester.run_loop_with_condition_test(train_data, model, condition,tsne=args.tsne, aggr=args.aggr)
   test_results_list = tester.run_loop_with_condition_test(test_data, model, condition,tsne=args.tsne, aggr=args.aggr)
   return train_results_list, test_results_list

def train_speaker_classifier(train_results_list):
   input_size = train_results_list[0]['predict'].shape[-1]
   nhu = 4096
   n_speakers = 8
   EPOCHS = 1000
   BATCH_SIZE = 1
   lr=0.0001

   sc_model = SpeakerClassifier(input_size, nhu, n_speakers)
   sc_model = sc_model.to(torch.device('cuda'))
   criterion = nn.CrossEntropyLoss()
   optimizer = torch.optim.Adam(sc_model.parameters(), lr=lr)

   sc_model.train()
   dl = torch.utils.data.DataLoader(train_results_list, batch_size=BATCH_SIZE, shuffle=True)
   print('training speaker classifier')
   for epoch in range(EPOCHS):
      # breakpoint()
      for batch in dl:
         optimizer.zero_grad()
         data = batch['predict'].to(torch.device('cuda')).squeeze(1).squeeze(1)
         wav_ids = batch['seq']
         speakers = ['_'.join(x.split('_')[:-2]) for x in wav_ids]
         targets = torch.LongTensor([spk_to_spkid[x] for x in speakers]).to(torch.device('cuda'))
         # data = torch.full(data.shape, targets.item()).to(torch.device('cuda'))/10 #can i learn the ints??
         #YES that one trains 
                           
         output = sc_model(data)
         targets = targets.repeat(output.shape[1])
         targets = targets.unsqueeze(0)
         output = output.reshape(output.shape[0], output.shape[2], output.shape[1]) #seq len last dim
         loss = criterion(output, targets)
         loss.backward()
         optimizer.step()
      print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}')
   return sc_model


def test_speaker_classifier(sc_model, test_results_list):
   #ensure that test_results_list contains train speakers only?
   # Testing the model
   sc_model.eval()
   correct = 0
   total = 0
   with torch.no_grad():
      for i in test_results_list:
         data = i['predict'].to(torch.device('cuda')).squeeze(0)
         wav_id = i['seq']
         speaker = '_'.join(wav_id.split('_')[:-2])
         target = torch.LongTensor([spk_to_spkid[speaker]]).to(torch.device('cuda'))
         outputs = sc_model(data) #prob across the N classes
         _, predicted = torch.max(outputs.data, 1)
         total += target.size(0)
         correct += (predicted == target).sum().item()
   acc = 100 * correct / total
   print(f'Accuracy: {acc:.2f}%')
   return acc

if __name__=='__main__':
   if args.experiment=='speaker_identification':
      main_model = load_main_model(args)
      train_data, test_data = get_data(args, main_model)
      sc_model = train_speaker_classifier(train_data)
      breakpoint()
      acc = test_speaker_classifier(sc_model, test_data)