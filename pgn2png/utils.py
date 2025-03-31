import chess
import chess.svg
import chess.pgn

import cairosvg
import re
import io

import os
import numpy as np
import json
import os
import numpy as np
from PIL import Image

from pathlib import Path

def is_valid_uci_move(move: str) -> bool:
  '''
  Check if the move is written in valid UCI syntax.
  
  Input:
    move: str -> the move to check
  Output:
    bool -> true if the move is valid, false otherwise
    
  '''
  if len(move) != 4:
    return False
  pattern = r'^[a-h][1-8][a-h][1-8]$'
  return bool(re.match(pattern, move))

def board2png(board: chess.Board, destination: str = 'image.png', lastmove: str = ''):
  '''
  Create a PNG image representing the board state.
  
  Input:
    board: chess.Board -> the board that will be represented as an image
    destination: str -> where to save the image (default is 'image.png')
    lastmove: str -> the last move made in the game (default is '')
  '''
  lastmove = chess.Move.from_uci(lastmove) if is_valid_uci_move(lastmove) else ''
  boardsvg = chess.svg.board(
          flipped=True,
          lastmove = lastmove,
          coordinates=False,
          board = board,
          size=40,
          colors={
              "square light": "#FFE4C4",
              "square dark": "#8B4513",
              "square dark lastmove": "#bdc959",
              "square light lastmove": "#f6f595"
          })
  f = open("position.svg", "w")
  f.write(boardsvg)
  f.close()
  cairosvg.svg2png(url='position.svg', write_to=destination, scale=7)

def extract_moves_from_pgn(pgn_string: str) -> list:
  '''
  Extract a list of moves from a PGN string of a chess game.
  
  Input:
    pgn_string: str -> the PGN of the game
    
  Output:
    list -> the list of moves in the game
  '''
  moves_list = []
  pgn = io.StringIO(pgn_string)
  game = chess.pgn.read_game(pgn)
  for move in game.mainline_moves():
      moves_list.append(move)
  return moves_list

def load_data_from_json(path: str) -> dict:
  '''
  Loads data from a JSON file.

  Input:
    path (str): The file path to the JSON file

  Output:
    dict: The parsed JSON data as a Python dictionary.
  '''
  with open(path, 'r') as openfile:
    return json.load(openfile)
  
def load_images_from_folder(folder: str) -> list:
  '''
  Loads all valid image files from a specified folder and converts them into NumPy arrays.
  
  Input:
      folder (str): The path to the folder containing image files.
  Output:
      list: A list of NumPy arrays representing the loaded images.
  '''
  images = []
  valid_image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp'}

  valid_images = [filename for filename in os.listdir(folder)
    if os.path.isfile(os.path.join(folder, filename)) and
    os.path.splitext(filename)[1].lower() in valid_image_extensions]

  valid_images.sort()

  for filename in valid_images:
    img_path = os.path.join(folder, filename)
    try:
      img = Image.open(img_path)
      img_array = np.array(img)   # Convert to numpy array
      images.append(img_array)    # Store the numpy array in the list
    except IOError:
      print(f"Cannot identify image file {img_path}")

  return images

def load_dataset_from_directory(base_folder: str) -> list:
  '''
  Load the dataset form a directory
  
  Input:
    base_folder: str -> the folder where there is the dataset
    
  Output:
    list -> the dataset loaded as a list
  '''
  instances = []
  for instance_folder in os.listdir(base_folder):
    instance_path = os.path.join(base_folder, instance_folder)
    if os.path.isdir(instance_path):
      data = load_data_from_json(os.path.join(instance_path, 'data.json'))
      images = load_images_from_folder(instance_path)
      instances.append({
        "site": data.get('site'),
        "opponent": data.get('opponent'),
        "is_white_master": data.get('is_white_master'),
        "result": data.get('result'),
        "game": data.get('game'),
        "images": images  # Store the actual image data
      })
  return instances



def make_dir(path: Path):
  '''
  Make a directory if not exist
  
  Input:
    path: Path -> the path of the directory
  '''
  try:
    path.mkdir(parents=True, exist_ok=True)
  except Exception as e:
    print(f"An error occurred: {e}")

def save_data_as_json(path: str, site: str, opponent: str, is_white_master: bool, result: str, game: str, index: int, opening: int, end_phase: int):
  '''
  Save the data as a JSON file
  
  Input:
    path: str
    site: str
    opponent: str
    is_white_master: bool
    result: str
    game: str,
    game_id: int,
    opening: int,
    end: int
  '''
  d =  {
    "site": site,
    "opponent": opponent,
    "is_white_master": is_white_master,
    "result": result,
    "game": game,
    "game_id": index,
    "opening": opening,
    "end_phase": end_phase,
  }
  json_obj = json.dumps(d, indent=4)
  with open(path + "/data.json", "w") as outfile:
    outfile.write(json_obj)




def get_move_number(img_name: str):
    pattern = r'\d+'

    return re.findall(pattern, img_name)

def extract_first_n_move(pgn_string: str, n: int, is_white: bool) -> str:
  semi_in_move = 2
  pgn = pgn_string.split(' ')
  ret = ''
  i = 0
  while (i < n*semi_in_move):
    if((is_white and i == (n*semi_in_move) - 1) or len(pgn) <= i):
       i = n*semi_in_move
       break
    ret += pgn[i] + ' '
    i += 1
  return ret