import csv
from enum import Enum


white_pieces = {
    'R': 5,
    'B': 3,
    'N': 3,
    'Q': 9,
    'P': 1
}

black_pieces = {
    'r': 5,
    'b': 3,
    'n': 3,
    'q': 9,
    'p': 1
}


def check_if_is_openings(fen: str) -> bool:
    '''
    Check if a FEN is in the openings csv file.
    
    Input:
        fen: str -> The FEN to analyze
    Output:
        bool -> True if the FEN is recognized as an opening, False otherwise.
        
    Example usage:
        is_opening = check_if_is_openings('rnbqkbnr/pppppp1p/8/6p1/6P1/8/PPPPPP1P/RNBQKBNR w KQkq - 0 2')
        print(is_opening)
    '''
    opening_file = 'openings/opening_fens.csv'
    data = []
    with open(opening_file, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(dict(row))
    for opening in data:
        if opening['FEN'] == fen:
            return True
    return False

def check_if_is_endgame(fen: str) -> bool:
    '''
    From a FEN, check if the position is considerable endgame. 
    The input fen could be the entire FEN (ex: "rn2qrk1/ppp4p/3b4/5p2/2NPpP2/P1P5/6PP/R1BQK2R w KQ - 1 15") 
    or only the pieces on the board (ex: "rn2qrk1/ppp4p/3b4/5p2/2NPpP2/P1P5/6PP/R1BQK2R").
    
    Input:
        fen: str -> the FEN of the position
    Output:
        bool -> True if both players have at most 13 points of material each, False otherwise
    '''
    w_material = 0
    b_material = 0
    for char in fen:
        if char == ' ':
            return (w_material <= 13) and (b_material <= 13) 
        try:
            if char.islower():
                b_material = b_material + black_pieces[char]
            elif char.isupper():
                w_material = w_material + white_pieces[char]
        except KeyError:
            continue
    return (w_material <= 13) and (b_material <= 13)

