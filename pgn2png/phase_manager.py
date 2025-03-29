import csv

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

