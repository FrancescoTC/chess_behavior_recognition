import re
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets
from huggingface_hub import login
import argparse

def extract_token():
    file_path = 'token.txt'
    with open(file_path, 'r') as f:
        content = f.read()
    
    match = re.search(r'"(.*?)"', content)
    if match:
        return match.group(1)
    else:
        return None

def save_split(dataset, phase):
    data = {'test': None, 'train': None}
    data['test'] = dataset['test'].filter(lambda e: e['phase'] == phase)
    data['train'] = dataset['train'].filter(lambda e: e['phase'] == phase)
    data = DatasetDict(data)
    data.save_to_disk("/workspace/datasets/" + phase)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--masters', nargs='+', required=True, help='Masters Name')
    args = parser.parse_args()
    
    dataset = {'test': None, 'train': None}
    
    
    names = args.masters
    
    for master in names:
        try:
            ds = load_dataset('ChessMoE/master_games_w_screenshots', master)

            def add_player(e):
                e['player'] = master
                return e
            
            ds['test'] = ds['test'].map(add_player)
            ds['train'] = ds['train'].map(add_player)
            
            if dataset['test'] == None:
                dataset['test'] = ds['test']
                dataset['train'] = ds['train']
            else:
                dataset['test'] = concatenate_datasets([dataset['test'], ds['test']])
                dataset['train'] = concatenate_datasets([dataset['train'], ds['train']])
                        
        except Exception:
            pass
    
    dataset = DatasetDict(dataset)
    save_split(dataset, 'opening')
    save_split(dataset, 'end')
    save_split(dataset, 'middle')
    
if __name__ == "__main__":
    login(token=extract_token())
    main()
