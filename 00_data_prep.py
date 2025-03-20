import gc

from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import login

'''
Divisione partite in base alla lunghezza
Considerazioni: Se una partita contiene solo le prime due mosse, molto spesso sono mosse molto comuni
                già dalla terza mossa in poi emerge una preferenza per alcune aperture basata sul tipo di giocatore
                La fase di apertura finisce in genere tra la 10 e la 15 mossa ma può arrivare anche alla mossa n.20
                Una partita arriva all'end game quando non ci sono più molti pezzi, il re non è in pericolo di matto
                e i pedoni diventano i protagonisti della partita. Questo accade in genere dopo le 35/40 mosse.
'''
min_len = 3
opening_phase = 15
end_game = 35

# Token di accesso per il dataset di HuggingFace
# token = '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'
# login(token=token)


# Carica in locale il dataset prima del training

# Carico il dataset
ds = load_dataset('FrancescoTC/chess_game')

print(ds)

# Divido i dati in 3 parti diverse in base alla durata delle partite.

# Opening Phase
dataset_opening = {'test': None, 'train': None}
dataset_opening['test'] = ds['test'].filter(lambda e: e["game_len"] <= opening_phase and e['game_len'] >= min_len)
dataset_opening['train'] = ds['train'].filter(lambda e: e["game_len"] <= opening_phase and e['game_len'] >= min_len)
dataset_opening = DatasetDict(dataset_opening)
dataset_opening.save_to_disk('/workspace/datasets/opening')

del dataset_opening
gc.collect()

# Middle game
dataset_middle = {'test': None, 'train': None}
dataset_middle['test'] = ds['test'].filter(lambda e: e["game_len"] > opening_phase and e["game_len"] < end_game)
dataset_middle['train'] = ds['train'].filter(lambda e: e["game_len"] > opening_phase and e["game_len"] < end_game)
dataset_middle = DatasetDict(dataset_middle)
dataset_middle.save_to_disk('/workspace/datasets/middle')
del dataset_middle
gc.collect()

# End game
dataset_end = {'test': None, 'train': None}
dataset_end['test'] = ds['test'].filter(lambda e: e["game_len"] >= end_game)
dataset_end['train'] = ds['train'].filter(lambda e: e["game_len"] >= end_game)
dataset_end = DatasetDict(dataset_end)
dataset_end.save_to_disk('/workspace/datasets/end')
del dataset_end
gc.collect()
