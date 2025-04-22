import chess, io, re

def remove_move_numbers(pgn_string) -> str:
    '''
    Having a pgn string in like "1.e4 e5 2..." or "1. e4 e5 2. ..."
    remove the move number and leave only the moves

    Inputs:
        pgn_string: str -> the PGN of the game
    Output:
        str -> the PGN of the game without move numbers
    '''
    period_index = pgn_string.find('.')
    if period_index != -1:
      return pgn_string[period_index + 1:]
    else:
      return pgn_string

def number_of_half_moves(pgn_string):
    '''
    From a pgn_string return the number of half moves in the game

    Inputs:
        pgn_string: str -> the PGN of the game
    Output:
        int -> the number of half moves in the game
    '''
    pgn_string = re.sub(r'[^a-zA-Z0-9.\s]', ' ', pgn_string)
    pgn_string = pgn_string.replace('\n', ' ')

    pgn_string = pgn_string.split(' ')
    pgn_string = [remove_move_numbers(s) for s in pgn_string]
    return [s for s in pgn_string if s != '']

  

def extract_moves_from_pgn(pgn_string: str) -> list:
    '''
    Extract a list of moves from a PGN string of a chess game.

    Input:
        pgn_string: str -> the PGN of the game
    
    Output:
        list -> the list of moves in the game
    Exceptions:
        None -> if the list of moves does not metch the moves in the pgn 
    '''
    moves_list = []
    pgn = io.StringIO(pgn_string)
    
    game = chess.pgn.read_game(pgn)
    
    for move in game.mainline_moves():
      moves_list.append(move)
    
    move_number_count = len(number_of_half_moves(pgn_string))
    
    if len(moves_list) != move_number_count:
      return None
    
    return moves_list


from chess.engine import PovScore

max_eval = 20_000 # mate
stockfish_path = "stockfish/stockfish-ubuntu-x86-64-avx2"
time_limit = 2.0

def score2number(score: PovScore):
    '''
    Having a stockfish evaluation, return a number that match the score

    Inputs:
        score: PovScore -> stockfish evaluation
    Output:
        int -> stockfish evaluation as an int
    '''
    if score.score() != None:
        return score.score()
    
    if score.is_mate():
        if score.mate() > 0:
            return max_eval if score.mate() == 0 else max_eval - 1000
        return -max_eval if score.mate() == 0 else max_eval + 1000

from chess import Board

def stockfish_eval_difference(board_0: Board, board_1: Board) -> int:
    '''
    Return the difference between the evaluation of the board_0 that represents
    the current board and the board_1 that represents the board_0 half move behind.

    Inputs:
        board_0: Board -> the current board
        board_1: Board -> the current board one half move behind
    Output:
        the diffenence between the stockfish evaluations
    '''
    with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
        result_0 = engine.analyse(board_0, chess.engine.Limit(time=time_limit))
        result_1 = engine.analyse(board_1, chess.engine.Limit(time=time_limit))

    score_0 = result_0['score'].relative
    score_1 = result_1['score'].relative

    numerical_score_0 = score2number(score_0) * -1
    numerical_score_1 = score2number(score_1)

    eval = round((numerical_score_0 - numerical_score_1) / 1000)
    
    print(f'n_actual: {numerical_score_0}\t-1: {numerical_score_1}')
    
    if eval >= 0:
        return eval if eval < 20 else 20
    
    return eval if eval > -19 else -19

import chess, chess.pgn

def evaluation_difference(pgn_str: str) -> int:
    '''
    For a given PGN, return the difference between the evaluation of the last 2 moves.
    
    Inputs:
        pgn_str: str -> the PGN of the game
    Output:
        int -> the difference of evaluation between the last 2 half moves
    '''
    current_board = chess.Board()
    moves = extract_moves_from_pgn(pgn_str)
    if moves == None:
        return -20
    
    len_game = len(moves)
    
    for index, move in enumerate(moves):
        if index == len_game - 1:
            halfmove_behind_board = current_board.copy()
        current_board.push(move)
    
    return stockfish_eval_difference(current_board, halfmove_behind_board)