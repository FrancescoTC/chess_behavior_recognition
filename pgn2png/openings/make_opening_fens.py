import csv
import chess
import sys, os

# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))
from utils import is_valid_uci_move, extract_moves_from_pgn

def get_fen(pgn: str) -> str:
    '''
    From a PGN extract the final FEN
    
    Input:
        pgn: str -> the pgn that we want to get the fen
    Output:
        the final position as FEN
    '''
    board = chess.Board()
    moves = extract_moves_from_pgn(pgn)
    
    for move in moves:
        if is_valid_uci_move(move.uci()):
            board.push(move=move)
    
    final_fen = board.fen()
    return final_fen

def make_fens(input_file: str, output_file: str):
    '''
    Make a TSV file that contain the FEN of the openings contained
    in a file as pgn.
    
    Input:
        input_file: str -> the name of the TSV file that contains the PGN of the openings
        output_file: str -> the name of the CSV file that will contain the FEN of the openings
    '''
    data = []
    print(input_file)
    with open(input_file, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter="\t")
        for row in reader:
            data.append(dict(row))
    
    for d in data:
        final_fen = get_fen(d['pgn'])

        with open(output_file, mode="a", newline="", encoding="utf-8") as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow([final_fen, input_file])

def get_final_move(pgn: str) -> int:
    moves = extract_moves_from_pgn(pgn)
    return len(moves)/2

def get_max_moves(input_file: str) -> int:
    max_move = 0
    name = ''
    data = []
    print(input_file)
    with open(input_file, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter="\t")
        for row in reader:
            data.append(dict(row))
            
    for d in data:
        final_move = get_final_move(d['pgn'])
        if final_move >= max_move:
            max_move = final_move
            name = d['name']
    return max_move, name