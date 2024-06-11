import os
import pandas as pd
import matplotlib.pyplot as plt
import sys
import re
import numpy as np

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

#### SPECIFIC EXPERIMENTS ####

#DP grid search 
#example path /Users/paz/vulture/home/paz/code/phd/FaceFormer/vocaset/opacus_grid_search/c_1_e_5_aggr_avg_ds_vertical_dp_opacus_delta0.01_nm_0.1_gn_1.0_tidx_-1_model_imitator_gen
def plot_dp_grid_search(folder):
    #plot train loss, vaild loss, accuracy and epsilon separately
    #also plot final epsilon: final train loss as a scatter plot?
    train_csv = 'client_0_train_results.csv'
    valid_csv = 'client_0_valid_results.csv'
    final_eps = []
    final_loss = []
    names = []
    for experiment in os.listdir(folder):
        if experiment.startswith('.'):
            continue
        if experiment.startswith('_'):
            continue
        if experiment.endswith('.png'):
            continue
        delta = experiment.split('delta')[1].split('_')[0]
        noise_multiplier = experiment.split('_')[12]
        max_grad_norm = experiment.split('_')[14]
        name = f'd-{delta} nm-{noise_multiplier} gn-{max_grad_norm}'
        exp_path = os.path.join(folder, experiment)
        train_results = pd.read_csv(os.path.join(exp_path, train_csv))
        valid_results = pd.read_csv(os.path.join(exp_path, valid_csv))
        plt.figure(1)
        plt.plot(train_results, label=name)
        plt.figure(2)
        plt.plot(valid_results['loss'], label=name)
        plt.figure(3)
        plt.plot(valid_results['accuracy'], label=name)
        plt.figure(4)
        plt.plot(valid_results['epsilon'], label=name)
        
        #for scatter plot, take last loss and last eps
        if float(noise_multiplier) >= 0.5:
            continue 
        final_eps.append(np.log(valid_results['epsilon'].iloc[-1].item()))
        final_loss.append(train_results.iloc[-1].item())
        names.append(name)


    plt.figure(1)
    plt.title('Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # plt.legend()
    #put legend outside chart
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(os.path.join(folder, 'train_results.png'))

    plt.figure(2)
    plt.title('Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # plt.legend()
    #put legend outside chart
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(os.path.join(folder, 'valid_loss.png'))

    plt.figure(3)
    plt.title('Valid Error')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    # plt.legend()
    #put legend outside chart
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(os.path.join(folder, 'valid_error.png'))

    plt.figure(4)
    plt.title('Train Eps')
    plt.xlabel('Epoch')
    plt.ylabel('Eps')
    # plt.legend()
    #put legend outside chart
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(os.path.join(folder, 'train_eps.png'))

    plt.figure(5)
    for i in range(len(final_loss)):
        plt.scatter(x=final_loss[i], y=final_eps[i], label=names[i])
    plt.title('Loss x eps')
    plt.xlabel('Loss')
    plt.ylabel('Eps')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(os.path.join(folder, 'loss_eps_scatter.png'))
    
if __name__=='__main__':
    # Example usage
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_paths", type=str, default='', help='path to folder containing model result csvs')
    parser.add_argument("--baseline_path", type=str, default='', help='path to folder containing baseline result csv')
    parser.add_argument("--dp", action='store_true', help='plot dp grid search instead')

    args = parser.parse_args()
    if args.dp:
        plot_dp_grid_search(args.folder_paths)
    else:
        plot_csv_data(args.folder_paths, args.baseline_path)
