#The sample hdtf files are too long to load in memory so i have to split them up but still conform to naming conventions

import os
import librosa 
import numpy as np
import math
import soundfile as sf
folder_path = '/mnt/nvme0n1p2/home/paz/data/phd/faceformer/hdtf'
vertices_folder = 'vertices_npy_orig'
wav_folder = 'wav_orig'
vertices_folder_out = 'vertices_npy'
wav_folder_out = 'wav'

max_len = 5 #try with 5 sec clips 
sampling_rate = 16000  # Sampling rate for audio
fps = 30  # Frames per second for vertex motion data
min_len = 1 #discard if last segment < 1 sec

def get_speaker_name(path):
    speaker_name = path.split('/')[-1].split('_sentence')[0]
    return speaker_name

# Process Numpy array files
def process():
    vertices_folder_path = os.path.join(folder_path, vertices_folder)
    new_vertices_folder_path = os.path.join(folder_path, vertices_folder_out)
    speaker_to_n_frames = {}
    for npyname in os.listdir(vertices_folder_path):
        if npyname.startswith('.'):
            continue

        npy_path = os.path.join(vertices_folder_path, npyname)
        vertex_data = np.load(npy_path, mmap_mode='r+')

        # Calculate the number of frames per segment
        frames_per_segment = max_len * fps

        # Calculate the number of segments
        total_frames = vertex_data.shape[0]
        num_segments = math.floor(total_frames / frames_per_segment)
        speaker = get_speaker_name(npyname)
        speaker_to_n_frames[speaker] = total_frames

        # Split and save each segment
        for i in range(num_segments):
            start_frame = i * frames_per_segment
            if i==num_segments:
                segment = vertex_data[start_frame:] #run till the end
            else:
                end_frame = min((i + 1) * frames_per_segment, total_frames)
                segment = vertex_data[start_frame:end_frame]
            # Create new filename
            base_name = os.path.splitext(npyname)[0]  # Remove extension
            new_filename = f"{base_name}{str(i + 1).zfill(2)}.npy"
            new_filepath = os.path.join(new_vertices_folder_path, new_filename)

            # Save the segment
            np.save(new_filepath, segment)

    wav_folder_path = os.path.join(folder_path, wav_folder)
    new_wav_folder_path = os.path.join(folder_path, wav_folder_out)
    for wavname in os.listdir(wav_folder_path):
        if wavname.startswith('.'):
            continue
        wav_path = os.path.join(wav_folder_path, wavname)
        speech_array, sampling_rate = librosa.load(wav_path, sr=16000)

        # Calculate the number of segments
        # total_seconds = len(speech_array) / sampling_rate
        speaker = get_speaker_name(wavname)
        total_seconds = speaker_to_n_frames[speaker]/fps

        num_segments = math.floor(total_seconds / max_len)

        # Split and save each segment
        for i in range(num_segments):
            start_frame = i * max_len * sampling_rate
            if i==num_segments:
                segment = speech_array[start_frame:]
            else:
                end_frame = min((i + 1) * max_len * sampling_rate, len(speech_array))
                segment = speech_array[start_frame:end_frame]
            # Create new filename
            base_name = os.path.splitext(wavname)[0]  # Remove extension
            new_filename = f"{base_name}{str(i + 1).zfill(2)}.wav"
            new_filepath = os.path.join(new_wav_folder_path, new_filename)

            # Save the segment
            sf.write(new_filepath, segment, sampling_rate)

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--process", action='store_true', help='process')
    parser.add_argument("--analyse", action='store_true', help='analyse')
    args = parser.parse_args()
    if args.process:
        process()
    elif args.analyse:
        hdtf_path = '/mnt/nvme0n1p2/home/paz/data/phd/faceformer/hdtf'
        hdtf_vertices_path = f'/mnt/nvme0n1p2/home/paz/data/phd/faceformer/hdtf/{vertices_folder_out}/'
        hdtf_wav_path = f'/mnt/nvme0n1p2/home/paz/data/phd/faceformer/hdtf/{wav_folder_out}/'
        vocaset_path = '/mnt/nvme0n1p2/home/paz/data/phd/faceformer/vocaset'
        vocaset_vertices_path = f'/mnt/nvme0n1p2/home/paz/data/phd/faceformer/vocaset/{vertices_folder_out}/'
        vocaset_wav_path = f'/mnt/nvme0n1p2/home/paz/data/phd/faceformer/vocaset/{wav_folder_out}/'

        one_vocaset_wav = vocaset_wav_path + os.listdir(vocaset_wav_path)[0]
        one_hdtf_wav = hdtf_wav_path + os.listdir(hdtf_wav_path)[0]
        speech_array_voca, sampling_rate = librosa.load(one_vocaset_wav, sr=16000)
        speech_array_hdtf, sampling_rate = librosa.load(one_hdtf_wav, sr=16000)

        one_vocaset_mesh = vocaset_vertices_path + os.listdir(vocaset_vertices_path)[0]
        one_hdtf_mesh = hdtf_vertices_path + os.listdir(hdtf_vertices_path)[0]
        vertices_voca = np.load(one_vocaset_mesh)
        vertices_hdtf = np.load(one_hdtf_mesh).reshape(-1, 15069)
