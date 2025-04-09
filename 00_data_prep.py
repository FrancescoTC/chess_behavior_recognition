import re
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import login

'''
Huggingface Token
'''

def extract_token():
    file_path = 'token.txt'
    with open(file_path, 'r') as f:
        content = f.read()
    
    match = re.search(r'"(.*?)"', content)
    if match:
        return match.group(1)
    else:
        return None


def get_move_number(pgn_string: str) -> int:
    '''
    Get the number of move contained into a PGN string.
    The PGN must has 2 strings separated by ' ' for each move.
    
    Input:
        pgn_string: str -> the PGN that rappresent the game.
    Output:
        int -> the number of move in the pgn
    '''
    semi_in_move = 2
    pgn = pgn_string.split(' ')
    return int(len(pgn) / semi_in_move)

def phase_generation(ds: Dataset, master: str, phase: str):
    '''
    Create a sub dataset that contain games that are from the same phase.
    
    Input:
        ds: Dataset -> the dataset that contain the game
        master: str -> the name of the master (subdir in ChessMoE/master_games_w_screenshots)
        phase: str -> the name of the phase (opening, middle, end)
    '''
    dataset = {'test': None, 'train': None}

    if phase == 'opening':
        filter_fn = lambda e: get_move_number(e["game"]) - 1 <= e['opening']
    elif phase == 'middle':
        filter_fn = lambda e: (get_move_number(e["game"]) - 1 > e['opening'] 
                               and get_move_number(e["game"]) - 1 < e['end_phase'])
    elif phase == 'end':
        filter_fn = lambda e: get_move_number(e["game"]) - 1 >= e['end_phase']
    else:
        raise ValueError(f"Unknown phase: {phase}")
    
    dataset['test'] = ds['test'].filter(filter_fn)
    dataset['train'] = ds['train'].filter(filter_fn)
    dataset.save_to_disk(f'/workspace/datasets/{phase}/{master}')

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--masters', nargs='+', required=True, help='Masters Name')
    args = parser.parse_args()
    
    names = args.names
    
    for master in names:
        try:
            ds = load_dataset('ChessMoE/master_games_w_screenshots', master)
            phase_generation(ds, master, 'opening')
            phase_generation(ds, master, 'middle')
            phase_generation(ds, master, 'end')
            
        except Exception:
            pass 

if __name__ == "__main__":
    login(token=extract_token())
    main()
