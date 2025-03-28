import utils as ig

from utils import get_move_number, extract_first_n_move

from pathlib import Path
import chess
from datasets import Dataset, DatasetDict
import os
import json
import shutil

def make_subdir_name(index: int, dir: str) -> str :
  '''
  Generate the subdir name that will keep an order. We will suppose that 
  all the master has at least 100_000 board states.
  
  Input:
    index: int -> the index of the subfolder
    dir: str -> the main directiory
    
  Output:
    the final path of the sub directory
  '''
  if index < 10:
    return dir + '/instance00000' + str(index)
  elif index < 100:
    return dir + '/instance0000' + str(index)
  elif index < 1000:
    return dir + '/instance000' + str(index)
  elif index < 10000:
    return dir + '/instance00' + str(index)
  elif index < 100000:
    return dir + '/instance0' + str(index)
  
  return dir + '/instance' + str(index)

def dataset_convertor(name: str, dir_name: str, ds: Dataset):
  '''
  Make the images dataset.
  
  Input:
    name: str -> the name of the master
    dir_name: str -> where we want to save the dataset
    ds: Dataset -> the original dataset
  '''
  
  ig.make_dir(Path(dir_name))
  index = 1
  print(name)

  for data in ds:
    print(f'{index} out of {len(ds)}')
    sub_dir = make_subdir_name(index, dir_name)
    ig.make_dir(Path(sub_dir))
    ig.save_data_as_json(sub_dir, data['site'], data['opponent'], data['is_white_master'], data['result'], data['game'], index)
    
    player_move = 0 if data['is_white_master'] else 1
    board = chess.Board()
    moves = ig.extract_moves_from_pgn(data['game'])
    move_index = 0

    for move in moves:
      if player_move == 0:
        move_index += 1
        m = str(board.fullmove_number) if board.fullmove_number > 9 else '0' + str(board.fullmove_number)
        img_name = sub_dir + '/move_' + m + '.png'
      
      board.push(move)
      if player_move == 0:
        ig.board2png(board=board, destination=img_name, lastmove=str(move))

      player_move = 1 if player_move == 0 else 0
    
    index += 1
    
def normalize_data(base_dir):
   """
   Normalize data into a flat structure with image paths and metadata.
   
   Input:
       base_dir (str): Base directory containing train and test folders.
   Output:
       list: A list of dictionaries containing image paths and metadata.
   """
   normalized_data = []

   for split in ['train', 'test']:
       split_path = os.path.join(base_dir, split)
       if not os.path.exists(split_path):
           continue
       
       for instance_folder in os.listdir(split_path):
           instance_path = os.path.join(split_path, instance_folder)
           
           if not os.path.isdir(instance_path):
               continue
           
           metadata_file = os.path.join(instance_path, 'data.json')
           if not os.path.exists(metadata_file):
               print(f"No metadata found in {instance_path}")
               continue
           
           with open(metadata_file, 'r') as f:
               metadata = json.load(f)
           
           for image_file in os.listdir(instance_path):
               if image_file.endswith('.png'):
                    move_number = get_move_number(image_file)[0]
                    image_path = os.path.join(instance_path, image_file)
                    game = extract_first_n_move(metadata["game"], int(move_number), metadata['is_white_master'])
                    normalized_data.append({
                        "image_path": image_path,
                        "site": metadata["site"].strip('"'),
                        "opponent": metadata["opponent"].strip('"'),
                        "is_white_master": metadata["is_white_master"],
                        "result": metadata["result"],
                        "game": game,
                        "split": split,
                        "game_id": instance_folder
                   })
                   
   return normalized_data

def create_flat_imagefolder(normalized_data, output_dir):
   """
   Create a train-test folder structure with images and metadata in a format compatible with load_dataset.
   
   Input:
       normalized_data (list): Normalized dataset with image paths and metadata.
       output_dir (str): Base directory where the dataset will be saved.
   """
   # Separate data by original splits
   train_data = [item for item in normalized_data if item['split'] == 'train']
   test_data = [item for item in normalized_data if item['split'] == 'test']
   
   splits = {'train': train_data, 'test': test_data}

   for split_name, split_data in splits.items():
       split_dir = os.path.join(output_dir, split_name)
       os.makedirs(split_dir, exist_ok=True)

       # Create metadata.jsonl first
       metadata_dict = {}
       for i, item in enumerate(split_data):
           image_name = f"{i}.png"
           metadata_dict[image_name] = {
               "site": item["site"],
               "opponent": item["opponent"],
               "is_white_master": item["is_white_master"],
               "result": item["result"],
               "game": item["game"], 
               "split": item['split'],
               "game_id": item['game_id']
           }
       
       # Save metadata.jsonl
       metadata_file = os.path.join(split_dir, "metadata.jsonl")
       with open(metadata_file, 'w') as f:
           for image_name, metadata in metadata_dict.items():
               # Include the image filename in the metadata
               metadata["file_name"] = image_name
               f.write(json.dumps(metadata) + '\n')
       
       # Copy images
       for i, item in enumerate(split_data):
           image_name = f"{i}.png"
           new_image_path = os.path.join(split_dir, image_name)
           shutil.copy(item["image_path"], new_image_path)

   print(f"Dataset created at {output_dir} with train-test split.")
   
def load_chess_dataset(data_dir):
   """
   Load the dataset with custom features, preserving splits.
   
   Input:
       data_dir (str): Directory containing the dataset.
   Output:
       DatasetDict: Hugging Face dataset dictionary with train and test splits.
   """
   features = Features({
       'image': Image(),
       'site': Value('string'),
       'opponent': Value('string'),
       'is_white_master': Value('bool'),
       'result': Value('string'),
       'game': Value('string')
   })

   dataset_dict = DatasetDict()
   
   for split in ['train', 'test']:
       split_path = os.path.join(data_dir, split)
       if os.path.exists(split_path):
           try:
               split_dataset = load_dataset(
                   "imagefolder",
                   data_dir=split_path,
                   features=features,
                   split=split
               )
               dataset_dict[split] = split_dataset
           except Exception as e:
               print(f"Error loading {split} split: {str(e)}")
               # Debug info
               print(f"Looking for metadata in: {split_path}")
               print(f"Files in directory: {os.listdir(split_path)}")
   
   return dataset_dict