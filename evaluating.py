import torch

import train

from datasets import Dataset

from unsloth import FastVisionModel # FastLanguageModel for LLMs

from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from transformers import TextStreamer

import os

def extract_player_name(output):
    # Split the output by spaces and get the last element
    return output.strip().split()[-1] 

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

def evaluate(dataset, model, tokenizer, file_dir):
    model.eval() # Set the model to evaluation mode
    correct, total = 0, 0
    text_streamer = TextStreamer(tokenizer, skip_prompt=True) # Create a TextStreamer for output streaming
    with torch.no_grad(): # Disable gradient calculation for evaluation
        for sample in dataset:  # Assuming dataset_test is your test dataset
            # Extract the image and other relevant information
            image = sample['image']
            conversation = convert_to_conversation(sample)
            instruction = conversation['messages'][0]['content'][0]['text']

            # Construct the messages
            messages = [
                {"role": "user", "content": [
                    {"type": "image"},  # Placeholder for the image
                    {"type": "text", "text": instruction}
                ]}
            ]

            # Prepare the input text using the tokenizer
            input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

            # Tokenize the inputs
            inputs = tokenizer(
                image,
                input_text,
                add_special_tokens=False,
                return_tensors="pt",
            ).to("cuda")  # Move to GPU if available
            # Generate predictions
            output = model.generate(
                **inputs,
                streamer=text_streamer,
                max_new_tokens=128,
                use_cache=True,
                temperature=1.5,
                min_p=0.1
            )

            # Process the output to get the predicted player name
            predicted_player = tokenizer.decode(output[0], skip_special_tokens=True)
            predicted_player = extract_player_name(predicted_player)
            print(predicted_player)
            # Compare the predicted player with the actual player
            actual_player = conversation['messages'][1]['content'][0]['text']  # Assuming this is the actual player name
            total += 1
            correct += (predicted_player == actual_player)  # Compare with the actual player
            print(f'\tActual Player:\t{actual_player}')
            print(f'\tcorrect/total:\t{correct}/{total}')

    # Calculate accuracy
    accuracy = correct / total if total > 0 else 0
    accuracy_text = f'Accuracy: {accuracy * 100:.2f}%'

    os.makedirs(file_dir, exist_ok=True)
    with open(file_dir + 'accuracy_output.txt', 'w') as file:
        file.write(accuracy_text)
