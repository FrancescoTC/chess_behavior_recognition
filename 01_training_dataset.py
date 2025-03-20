from datasets import Dataset, DatasetDict
from unsloth import UnslothVisionDataCollator, FastVisionModel, is_bf16_supported
from trl import SFTTrainer, SFTConfig
import sys
import io
import os

from train import DynamicDataset, convert_to_conversation

game_state = 'end'
dataset_location = '/workspace/datasets/' + game_state

print("Loading the dataset")
dataset = {'test': None, 'train': None}
dataset['test'] = Dataset.load_from_disk(dataset_location + '/test')
print(f'Dataset test loaded:\n{dataset}\n')
dataset['train'] = Dataset.load_from_disk(dataset_location + '/train')
print(f'Dataset train loaded:\n{dataset}\n')
dataset = DatasetDict(dataset)
ds = dataset['train']

path = '/workspace/model/' + game_state + '/'

print("loading unsloth/Llama-3.2") 
model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Llama-3.2-11B-Vision-Instruct",
    load_in_4bit = True, # Use 4bit to reduce memory use. False for 16bit LoRA.
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
)

model = FastVisionModel.get_peft_model(
    model,
    
    finetune_vision_layers     = True,  # Fine-tunes the visual backbone
    finetune_language_layers   = True,  # Fine-tunes the text-processing layers
    finetune_attention_modules = True,  # Fine-tunes attention mechanisms
    finetune_mlp_modules       = True,  # Fine-tunes Multi-Layer Perceptron (MLP) modules in the model
    
    r = 16,             # The rank of the LoRA adaptation
    lora_alpha = 32,    # Recommended alpha == r at least or 2x
    lora_dropout = 0.1, # Add dropout for regularization
    bias = "none",      # No bias in lora layers
    random_state = 3407,# Random seed for reproducibility
    use_rslora = True,  # Rank-Stabilized LoRA (RS-LoRA), which improves LoRA's performance in large-scale models by stabilizing low-rank decompositions
    loftq_config = None,# LoRA for Quantized Training
)

dynamic_dataset = DynamicDataset(ds)

print("Model setted to training mode")
FastVisionModel.for_training(model) # Enable for training!

print("Satting hyperparameter for training")
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = UnslothVisionDataCollator(model, tokenizer),
    train_dataset = dynamic_dataset,
    args = SFTConfig(
        per_device_train_batch_size = 2,    # each GPUs processes 2 samples per step
        gradient_accumulation_steps = 4,    # combines gradients from 4 batches before updating weights
        warmup_steps = 100,                 # Warmup steps for stability
        max_steps = 2000,                   # Total optimization steps
        num_train_epochs = 3,
        learning_rate = 5e-5,               # Learning rate
        fp16 = not is_bf16_supported(),     # Uses 16-bit floating point (FP16) of BF16 is not supported
        bf16 = is_bf16_supported(),         # Uses BFloat16 (BF16) if supported
        logging_steps = 10,                 # Logs metrics (loss, accuracy, etc.) every 10 steps
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine_with_restarts",  # Better learning rate decay
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        dataset_num_proc = 4,
        max_seq_length = 2048,
    ),
    formatting_func=convert_to_conversation,
)

print("start training")
trainer.train()

print("saving model and tokenizer")
model.save_pretrained(path + "/lora_model")
tokenizer.save_pretrained(path + "/lora_model")
