import os 
import librosa 

import argparse as ap
parser = ap.ArgumentParser()
parser.add_argument('data_path', type=str, help='path to dataset to eval')
parser.add_argument('--dataset', type=str, default='vocaset', help='which dataset to eval')
parser.add_argument('--save', action='store_true', help='save to csv or no')
args = parser.parse_args()

wavs_path = f'{args.data_path}/wav'
verts_path = f'{args.data_path}/vertices_npy'
template_path = f'{args.data_path}/templates.pkl'

wavs = os.listdir(wavs_path) #speakeri_sentenceXX.wav ...
verts = os.listdir() #speakeri_sentenceXX.npy ...

print('getting wav durations')
spk_to_wav = {} #dict: {speaker: {wav_id: duration}}
for wav in wavs:
    if wav.startswith('.'):
        continue
    wav_path = os.path.join(wavs_path, wav)
    speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
    duration = speech_array.shape[0]/sampling_rate
    speaker = '_'.join(wav.split('_')[:-1])
    wav = wav.split('_')[-1]
    if speaker in spk_to_wav:
        spk_to_wav[speaker][wav] = duration
    else:
        spk_to_wav[speaker] = {wav: duration} 

wniv = set([x.split('.')[0] for x in verts]) - set([x.split('.')[0] for x in wavs])
vniw = set([x.split('.')[0] for x in wavs]) - set([x.split('.')[0] for x in verts])
print('wavs not in verts', wniv)
print('verts not in wavs', vniw)

total_dur = 0
total_wavs = 0
for speaker in spk_to_wav:
    speaker_dur = 0
    n_speaker_wavs = 0
    for wav in spk_to_wav[speaker]:
        duration = spk_to_wav[speaker][wav]
        speaker_dur += duration
        total_dur += duration
        total_wavs +=1
        n_speaker_wavs +=1
    print(f'speaker: {speaker}, dur: {speaker_dur:0.2f}, n: {n_speaker_wavs}')

print(f'total duration: {total_dur:0.2f}, n: {total_wavs}')


# input_values = np.squeeze(processor(speech_array,sampling_rate=16000).input_values)
# key = f.replace("wav", "npy")
# data[key]["audio"] = input_values

# if args.dataset == "vocaset":
#     data[key]["vertice"] = np.load(vertice_path,allow_pickle=True)[::2,:]#due to the memory limit
# elif args.dataset == "hdtf":
#     data[key]["vertice"] = np.load(vertice_path,allow_pickle=True)[::2,:].reshape(-1, int(args.vertice_dim))
# elif args.dataset == "BIWI":
#     data[key]["vertice"] = np.load(vertice_path,allow_pickle=True)

# if __name__=='__main__':

