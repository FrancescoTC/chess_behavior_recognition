import os
from datasets import load_dataset, DatasetDict
from huggingface_hub import login

'''
Huggingface toker or credential
'''


def get_subdirectories(directory):
    try:
        subdirs = []
        
        for item in os.listdir(directory):

            full_path = os.path.join(directory, item)

            if os.path.isdir(full_path):
                subdirs.append(item)
        
        return subdirs
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


if __name__ == "__main__":
    '''
    Example of usage of this script:
        dir_name = images_datasets (directory where there are all the datasets that you want
        to load divided in subdirectory with the name of the chess player)
        
        load_dataset_to_huggingface.py images_datasets\n
    '''
    dir_name = input("Enter the directory name: ")
    subdirectories = get_subdirectories(dir_name)
    for name in subdirectories:
        print(name)
        ds = {'train': None, 'test': None}
        print('train')
        ds['train'] = load_dataset(dir_name + "/" + name + "/train/")
        ds['train'] = ds['train']['train']
        print('test')
        ds['test'] = load_dataset(dir_name + "/" + name + "/test/")
        ds['test'] = ds['test']['train']
        ds = DatasetDict(ds)
        ds.push_to_hub("ChessMoE/master_games_w_screenshots", name)