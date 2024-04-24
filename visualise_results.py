import os
import pandas as pd
import matplotlib.pyplot as plt
import sys
import re

def do_plot(folder_path, file, is_multi, fig_id):
    file_path = os.path.join(folder_path, file)
    # Read CSV file
    data = pd.read_csv(file_path)
    if is_multi:
        label = 'train_idx_'
        label += folder_path.split('train_idx_')[1][0]
    else:
        label = file.split('_results.csv')[0]
    # Plot loss
    plt.figure(fig_id)
    plt.plot(data['loss'], label=label)
    if 'accuracy' in data:
        # Plot accuracy
        plt.figure(fig_id+1)
        plt.plot(data['accuracy'], label=label)

def plot_csv_data(folder_paths, baseline_path=''):
    # Iterate over files in the folder
    is_multi = False
    has_train_results = False
    if len(folder_paths.split())>1:
        print('testing multiple models!')
        is_multi=True
    for folder_path in folder_paths.split():
        folder_path = re.sub('/Users/paz/vulture/home/', '/home/', folder_path)
        print(f'plotting csvs in {folder_path}')
        for file in os.listdir(folder_path):
            if not file.endswith('.csv'):
                continue
            if file.startswith('.'):
                continue
            if file.endswith('_train_results.csv'):
                print('has train results')
                has_train_results = True
                do_plot(folder_path, file, is_multi, 3)
                continue
            do_plot(folder_path, file, is_multi, 1)

        if args.baseline_path:
            for file in os.listdir(baseline_path):
                if not file.endswith('.csv'):
                    continue
                if file.startswith('.'):
                    continue
                if file.endswith('_train_results.csv'):
                    print('has train results')
                    has_train_results = True
                    do_plot(folder_path, file, is_multi, 3)
                    continue
                do_plot(folder_path, file, is_multi, 1)

    # Loss graph settings
    plt.figure(1)
    plt.title('Loss per Client')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    # plt.legend()
    #put legend outside chart
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, 'valid_loss.png'))

    # Accuracy graph settings
    plt.figure(2)
    plt.title('Error per Client')
    plt.xlabel('Round')
    plt.ylabel('Error')
    # plt.legend()
    #put legend outside chart
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, 'valid_accuracy.png'))
    plt.close()
    if has_train_results:
        plt.figure(3)
        plt.title('Loss per Client')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        # plt.legend()
        #put legend outside chart
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig(os.path.join(folder_path, 'train_loss.png'))

if __name__=='__main__':
    # Example usage
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_paths", type=str, default='', help='path to folder containing model result csvs')
    parser.add_argument("--baseline_path", type=str, default='', help='path to folder containing baseline result csv')
    args = parser.parse_args()
    
    plot_csv_data(args.folder_paths, args.baseline_path)
