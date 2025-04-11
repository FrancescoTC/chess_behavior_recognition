from enum import Enum
from datasets import Dataset, concatenate_datasets
import csv

masters_name_file = "masters_name.csv"

class Split(Enum):
    '''
    Enum for defining the test or the train split
    Example usage
    print(Split.TRAIN)          # Output: Split.TRAIN
    print(Split.TRAIN.value)    # Output: "train"
    
    '''
    TRAIN = "train"
    TEST = "test"

# def levels_to_smallest(dataset_location: str, split: Split):
#     dataset_location = dataset_location + '/' + split.value
#     ds = Dataset.load_from_disk(dataset_location)
#     ds = ds.shuffle()
#     return ds.select(range(10))

def get_masters_name():
    data = []
    print(masters_name_file)
    with open(masters_name_file, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(dict(row))
            
    return data

def levels_to_smallest(dataset_location: str, split: Split):
    '''
    Function that take a location where is saved the dataset containing all the game
    of every masters and return a new dataset where every masters have the same ammount of
    game
    Input:
        dataset_location: str -> where to take the initial dataset
        split: Split -> if it's the training or the test split
    Output:
    '''
    
    print("Loading the " + split.value + " split of the dataset...")
    dataset_location = dataset_location + '/' + split.value
    ds = Dataset.load_from_disk(dataset_location)
    ds = ds.shuffle()
    print(ds)

    # Take only the smallest number of sample for each masters
    masters = get_masters_name()
    masters_ds = [None] * len(masters)
    final_ds = None  # Initialize an empty dataset

    min_len = float("inf")  # Initialize with a very large number

    for i in range(len(masters)):
        masters_ds[i] = ds.filter(lambda x: x["player"] == masters[i]['name'])
        l = masters_ds[i].num_rows 
        if l > 0:
            min_len = min(min_len, l)  # Update min_len with the smallest dataset size
        

    print(f"Only {min_len} samples for each master will be taken")  # Print the smallest dataset size found


    for i in range(len(masters)):
        if final_ds is None and len(masters_ds[i]) != 0:
            final_ds = masters_ds[i].select(range(min_len))  # First dataset, initialize final_ds
        else:
            if len(masters_ds[i]) != 0:
                final_ds = concatenate_datasets([final_ds, masters_ds[i].select(range(min_len))])  # Concatenate datasets
    
    return final_ds.select(range(10))