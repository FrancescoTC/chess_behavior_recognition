from datasets import Dataset
from unsloth import UnslothVisionDataCollator, FastVisionModel, is_bf16_supported
from trl import SFTTrainer, SFTConfig
import base64
import io
import os

def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def convert_to_conversation(sample):
    instruction = "You are an expert chess player. Who is playing this game in the role of " + ("white" if sample['is_white_master'] else "black") + \
        ". The image that you seen is a screenshot of the game: " + sample['game'] + ". " + \
        "Please choose from the following options: Carlsen, Nakamura, Morphy, Kasparov, Capablanca, Fischer"
    conversation = [
        { "role": "user",
          "content" : [
            {"type" : "text",  "text"  : instruction},
            {"type" : "image", "image" : sample["image"]} ]
        },
        { "role" : "assistant",
          "content" : [
            {"type" : "text",  "text"  : sample["player"]} ]
        },
    ]
    return { "messages" : conversation }
pass

class DynamicDataset:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        return convert_to_conversation(self.dataset[idx])

    def __len__(self):
        return len(self.dataset)