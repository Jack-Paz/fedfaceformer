import os
import pandas as pd
import matplotlib.pyplot as plt
import sys

def plot_csv_data(folder_path):
    # Iterate over files in the folder
    print(f'plotting csvs in {folder_path}')
    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            if file.startswith('.'):
                continue
            file_path = os.path.join(folder_path, file)
            # Read CSV file
            data = pd.read_csv(file_path)
            label = file.split('_results.csv')[0]
            # Plot loss
            plt.figure(1)
            plt.plot(data['loss'], label=label)

            # Plot accuracy
            plt.figure(2)
            plt.plot(data['accuracy'], label=label)

    # Loss graph settings
    plt.figure(1)
    plt.title('Loss per Client')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    # plt.legend()
    #put legend outside chart
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, 'loss.png'))

    # Accuracy graph settings
    plt.figure(2)
    plt.title('Error per Client')
    plt.xlabel('Round')
    plt.ylabel('Error')
    # plt.legend()
    #put legend outside chart
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, 'accuracy.png'))
    plt.close()

if __name__=='__main__':
    # Example usage
    folder_path = sys.argv[1]
    
    plot_csv_data(folder_path)
