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



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--masters', nargs='+', required=True, help='Masters Name')
    args = parser.parse_args()
    
    dataset = {'test': Dataset(), 'train': Dataset()}
    
    
    names = args.names
    
    for master in names:
        try:
            ds = load_dataset('ChessMoE/master_games_w_screenshots', master)

            def add_player(e):
                example['player'] = master
                return e
            
            ds['test'] = ds['test'].map(add_player_feature)
            ds['train'] = ds['train'].map(add_player_feature)
            
            dataset['test'] = concatenate_datasets([dataset['test'], ds['test']])
            dataset['train'] = concatenate_datasets([dataset['train'], ds['train']])
                        
        except Exception:
            pass
        
    dataset.save_to_disk("/workspace/dataset")

if __name__ == "__main__":
    login(token=extract_token())
    main()
