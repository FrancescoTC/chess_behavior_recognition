from unsloth import FastVisionModel
from datasets import Dataset, concatenate_datasets
import evaluating

dataset_location = "middle"
print("Loading the dataset")
dataset = Dataset.load_from_disk('/workspace/datasets/middle/test')

ds1 = dataset.filter(lambda example: example['player'] == 'Nakamura')
ds1 = ds1.select(range(7700))

ds2 = dataset.filter(lambda example: example['player'] == 'Capablanca')

model, tokenizer = FastVisionModel.from_pretrained(
    model_name = "model/middle/lora_model", # YOUR MODEL YOU USED FOR TRAINING
    load_in_4bit = True, # Set to False for 16bit LoRA
)
FastVisionModel.for_inference(model)
evaluating.evaluate(ds1, model, tokenizer, '/workspace/models/middle/')
evaluating.evaluate(ds2, model, tokenizer, '/workspace/models/middle/')
