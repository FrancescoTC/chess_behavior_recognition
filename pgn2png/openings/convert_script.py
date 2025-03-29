import argparse
from make_opening_fens import make_fens


if __name__ == "__main__":
    
    # Take in input a list of TSV files 
    parser = argparse.ArgumentParser(description="Process a list of arguments.")
    parser.add_argument('arguments', nargs='+', help='The names of the TSV file without the .tsv extension')
    args = parser.parse_args()
    
    # for each TSV file put all the final fen of the PGN in a single file 
    output_file = 'opening_fens.tsv'
    for arg in args.arguments:
        make_fens(arg, output_file)
