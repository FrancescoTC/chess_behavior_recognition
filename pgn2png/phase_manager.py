import csv
from enum import Enum


pieces = {
    'R': 5,
    'B': 3,
    'N': 3,
    'Q': 9
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

def get_non_pawn_materials(fen: str) -> int:
    '''
    From a FEN get the value of the non-pawn materials. 
    The input fen could be the entire FEN (ex: "rn2qrk1/ppp4p/3b4/5p2/2NPpP2/P1P5/6PP/R1BQK2R w KQ - 1 15") 
    or only the pieces on the board (ex: "rn2qrk1/ppp4p/3b4/5p2/2NPpP2/P1P5/6PP/R1BQK2R").
    
    Input:
        fen: str -> the FEN of the position
    Output:
        int -> the materials value of the non-pawn pieces
    '''
    material = 0
    for char in fen:
        if char == ' ':
            return material
        try:
            material = material + pieces[char.upper()]
        except KeyError:
            continue
    return material 

