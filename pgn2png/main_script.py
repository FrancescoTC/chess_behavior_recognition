from datasets import load_dataset
from huggingface_hub import login

'''
Huggingface toker or credential
'''

token = '...'
login(token=token)

ds_morphy = load_dataset("ChessMoE/games", "Morphy")

from dataset_generator import dataset_convertor

master_name = 'Morphy'
image_dir = 'data/Morphy/'

'''
Prendo come esempio le partite di Morphy e per semplicità considero solo
le prime 10.
La divisione verrà comunque fatta in train e test set
'''
dataset = ds_morphy['train'].select(range(10))
dataset_convertor(master_name, image_dir + '/train', dataset)

dataset = ds_morphy['test'].select(range(10))
dataset_convertor(master_name, image_dir + '/test', dataset)

'''
Qui ho delle cartelle contenente tutti gli screenshot delle partite giocate da morphy.
Ora voglio trasformare quelle cartelle in un unico dataset di immagini che poi potrò usare per il training dei modelli
'''

from dataset_generator import normalize_data, create_flat_imagefolder, load_chess_dataset

normalized_dataset = normalize_data(image_dir)
# Create train-test structure
output_directory = "imagefolder_dataset/" + master_name
create_flat_imagefolder(normalized_dataset, output_directory)