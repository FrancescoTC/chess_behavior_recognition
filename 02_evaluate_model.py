from unsloth import FastVisionModel
from datasets import Dataset, concatenate_datasets
import evaluating

dataset_location = "middle"
print("Loading the dataset")
dataset = Dataset.load_from_disk('/workspace/datasets/' + dataset_location + '/test')

ds = dataset

model, tokenizer = FastVisionModel.from_pretrained(
    model_name = "model/middle/Qwen/lora_model", # YOUR MODEL YOU USED FOR TRAINING
    load_in_4bit = True, # Set to False for 16bit LoRA
)
FastVisionModel.for_inference(model)
evaluating.evaluate(ds, model, tokenizer, '/workspace/' + dataset_location + '/middle/', 'mistral')

# Carlsen
# 
# ds = dataset
# 
# ds = ds.filter(lambda x: x["player"] == "Carlsen")[:10000]
# ds = Dataset.from_dict(ds)
# 
# model, tokenizer = FastVisionModel.from_pretrained(
#     model_name = "model/" + dataset_location + "/lora_model", # YOUR MODEL YOU USED FOR TRAINING
#     load_in_4bit = True, # Set to False for 16bit LoRA
# )
# FastVisionModel.for_inference(model)
# evaluating.evaluate(ds, model, tokenizer, '/workspace/' + dataset_location + '/middle/', 'mistral')
# 
# # Gli altri
# 
# ds = dataset
# 
# ds = ds.filter(lambda x: x["player"] != "Carlsen" and x['player'] != 'Nakamura')
# 
# model, tokenizer = FastVisionModel.from_pretrained(
#     model_name = "model/" + dataset_location + "/lora_model", # YOUR MODEL YOU USED FOR TRAINING
#     load_in_4bit = True, # Set to False for 16bit LoRA
# )
# FastVisionModel.for_inference(model)
# evaluating.evaluate(ds, model, tokenizer, '/workspace/' + dataset_location + '/middle/', 'mistral')
# 