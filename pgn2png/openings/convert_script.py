import argparse
from make_opening_fens import make_fens, get_max_moves


if __name__ == "__main__":
    
    # Take in input a list of TSV files 
    parser = argparse.ArgumentParser(description="Process a list of arguments.")
    parser.add_argument('arguments', nargs='+', help='The names of the TSV file')
    args = parser.parse_args()
    
    # for each TSV file, put all the final FEN of the PGN in a single file 
    output_file = 'opening_fens.csv'
    for arg in args.arguments:
        make_fens(arg, output_file)
        max_move, opening_name = get_max_moves(arg)
        print(f'For the file {arg}, the longest opening (or at least one of the longest) is {opening_name}, with {max_move} moves.')
