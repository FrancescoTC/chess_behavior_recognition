import torch
import os
import re

from transformers import PreTrainedModel, PretrainedConfig 
from transformers.modeling_outputs import ModelOutput
from contextlib import nullcontext
from safetensors import safe_open
import json
import yaml

from gpt_model import GPT, GPTConfig
from chess_tokenizer import ChessTokenizer


def get_ckpt_path(directory):
    """
    Get the number of the latest(highest) checkpoint in the given directory.
    """
    ckpt_numbers = [
        int(path.split("_")[1].split(".")[0])
        for path in os.listdir(directory)
        if path.startswith("ckpt_") and path.endswith(".pt")
    ]  # all of the saved checkpoint iteration numbers
    assert len(ckpt_numbers) > 0, f"No checkpoints found in {directory}"
    return sorted(ckpt_numbers)[-1]

def get_safetensors_path(pretrained_model_path):
        """
        Get the path to the .safetensors file in the given directory.
        """
        checkpoint_path = None
        for file in os.listdir(pretrained_model_path):
            if file.endswith(".safetensors"):
                checkpoint_path = os.path.join(pretrained_model_path, file)
                break
        if checkpoint_path is None:
            raise FileNotFoundError(f"No .safetensors file found in {pretrained_model_path}")
        return checkpoint_path

def search_for_config(pretrained_model_path: str) -> PretrainedConfig:
    """
    Search for the configuration file in the given directory and return the configuration object.
    """
    if not os.path.isdir(pretrained_model_path):
        config_path = os.path.dirname(pretrained_model_path)
        for file in os.listdir(config_path):
            if file.endswith(".yaml"):
                # Load the configuration from the yaml file
                with open(os.path.join(config_path, file), "r") as file:
                    config_dict = yaml.safe_load(file)

                # Create the configuration object from the dictionary
                return PretrainedConfig(**config_dict)
        # If no yaml file is found, raise an error
        raise FileNotFoundError(f"No configuration file found in {pretrained_model_path}")
    else:
        return PretrainedConfig.from_pretrained(pretrained_model_path)

# These are the model args that should not be changed from the checkpoint
fixed_model_args = ["n_layer", "n_head", "n_embd", "vocab_size", "bias"]


def fix_state_dict_keys(state_dict):
    """
    Fix the keys of the state dictionary, removing the unwanted prefix '_orig_mod.' that sometimes appears from the checkpoint.
    """
    unwanted_prefix = "_orig_mod."
    unwanted_prefix_2 = "model."
    for k, _ in list(state_dict.items()):
        fix = k
        if fix.startswith(unwanted_prefix):
            fix = fix[len(unwanted_prefix) :]
        if fix.startswith(unwanted_prefix_2):
            fix = fix[len(unwanted_prefix_2) :]
        if ".experts." in fix:
            fix = re.sub(r"\.experts\.\d+", "", fix)

        if fix != k:
            print(f"Fixing key {k} to {fix}")
            state_dict[fix] = state_dict.pop(k)
    return state_dict


class CustomGPTModel(PreTrainedModel):
    def __init__(self, config, checkpoint: dict):
        super().__init__(config)

        # Create the model from the config
        model = GPT(
            GPTConfig(
                n_layer=config.n_layer,
                n_head=config.n_head,
                n_embd=config.n_embd,
                block_size=config.block_size,
                vocab_size=config.vocab_size,
                dropout=config.dropout,
                bias=config.bias
            )
        )

        self.device_type = "cuda" if "cuda" in config.device else "cpu"
        self.ptdtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[config.dtype]

        print(f"Loading model on {self.device_type} with dtype {self.ptdtype}")

        self.ctx = (
            nullcontext()
            if self.device_type == "cpu"
            else torch.amp.autocast(device_type=self.device_type, dtype=self.ptdtype)
        )
        
        # Load the model state dict from the checkpoint and fix the keys
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        state_dict = fix_state_dict_keys(checkpoint["model"])

        # Load the state dict into the model
        model.load_state_dict(state_dict)

        # Set the iteration number from the checkpoint
        self.iter_num: int = checkpoint.get("iter_num", 0)
        self.origin_file:str = checkpoint.get("origin_file", "")

        # Create the optimizer
        self.optimizer = model.configure_optimizers(
            weight_decay=config.weight_decay,
            learning_rate=config.learning_rate,
            betas=(config.beta1, config.beta2),
            device_type=config.device,
        )
        if checkpoint["optimizer"] is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        # Save the model
        self.model = model

    @torch.enable_grad()
    def forward(self, input_ids: torch.tensor, targets:torch.tensor=None , attention_mask=None, labels=None, *pos_args, **kwargs) -> ModelOutput:
        # # Input without the last token
        # X = input_ids[:, :-1]

        # # Target without the first token
        # Y = input_ids[:, 1:]

        with self.ctx:
            # GPT forward doesn't want other arguments
            logits, loss = self.model(idx=input_ids, targets=targets)

        return ModelOutput(loss=loss, logits=logits)

    @classmethod
    def from_pretrained_safe_tensors(cls, pretrained_model_path: str, config: PretrainedConfig, *model_args, **kwargs):
        print(f"Searching safetensors in {pretrained_model_path}")
        # If temperature is set to a value less than or equal to 1, enable sampling
        if config.temperature is not None and config.temperature <= 1:
            config.do_sample = True

        # Search for the .safetensors file
        checkpoint_path = get_safetensors_path(pretrained_model_path)
        print(f"Loading checkpoint safetensors from {pretrained_model_path}")
        
        # Load the state_dict from .safetensors file
        state_dict = {}
        with safe_open(checkpoint_path, framework="pt", device=config.device) as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)

        # Extract model arguments from `config.json` file
        config_path = os.path.join(pretrained_model_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config_data: dict = json.load(f)
        else:
            FileNotFoundError(f"No config.json file found in {pretrained_model_path}")

        # Apply fixed model args (if any) to config
        for arg in fixed_model_args:
            if arg in config_data:
                setattr(config, arg, config_data[arg])

        # Handle block size configuration
        ckpt_block_size = config.block_size if hasattr(config, 'block_size') else config_data.get("block_size", None)
        required_block_size = config.block_size if config.block_size < ckpt_block_size else None
        setattr(config, "block_size", ckpt_block_size)

        # Create the checkpoint dictionary
        checkpoint = {
            "model": state_dict,
            "iter_num": config_data.get("iter_num", 0),
            "optimizer": config_data.get("optimizer", None),
            "origin_file": checkpoint_path,
        }

        # Initialize the model with config and state_dict from .safetensors
        model = CustomGPTModel(config, checkpoint)

        # Eventually crop the model's block size
        if required_block_size is not None:
            print(f"Cropping model block size from {ckpt_block_size} to {required_block_size}")
            model.model.crop_block_size(required_block_size)
            setattr(config, "block_size", required_block_size)

        # # Additional optional loads, e.g., special tokens or vocab
        # special_tokens_path = os.path.join(pretrained_model_path, "special_tokens_map.json")
        # if os.path.exists(special_tokens_path):
        #     with open(special_tokens_path, "r") as f:
        #         model.special_tokens_map = json.load(f)

        # tokenizer_config_path = os.path.join(pretrained_model_path, "tokenizer_config.json")
        # if os.path.exists(tokenizer_config_path):
        #     with open(tokenizer_config_path, "r") as f:
        #         model.tokenizer_config = json.load(f)

        # vocab_path = os.path.join(pretrained_model_path, "vocab.json")
        # if os.path.exists(vocab_path):
        #     with open(vocab_path, "r") as f:
        #         model.vocab = json.load(f)

        return model

    def from_pretrained_pt(pretrained_model_path: str, config: PretrainedConfig, *model_args, **kwargs):
        print(f"Searching model in", pretrained_model_path)
        # Load checkpoint number from parameters or search for the highest one in the directory
        if pretrained_model_path.endswith(".pt"):
            checkpoint_path = pretrained_model_path
        else:    
            n_checkpoint = kwargs.pop('n_checkpoint', None)#, get_ckpt_path(pretrained_model_path))
            if n_checkpoint is None:
                n_checkpoint = get_ckpt_path(pretrained_model_path)

            # Check checkpoint number value
            if (not isinstance(n_checkpoint, int)) and (not isinstance(n_checkpoint, str)):
                raise ValueError(f"n_checkpoint must be an integer or a string, not {type(n_checkpoint)}")

            ckpt = f"ckpt_{n_checkpoint}.pt"
            checkpoint_path = os.path.join(pretrained_model_path, ckpt)

        # Define the checkpoint path
        print(f"Loading from", checkpoint_path)

        # Print the model type
        print(f"\t model of type: {config.model_type}")

        # If temperature is set to a value less than or equal to 1, enable sampling
        if config.temperature is not None and config.temperature <=1:
            config.do_sample = True

        # Load the checkpoint weights from '.pt' file
        checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=False)

        # Save the origin file path
        checkpoint["origin_file"] = checkpoint_path

        # Extract the model args from the checkpoint
        checkpoint_model_args = checkpoint["model_args"]

        # Overwrite the model_args from the config with the ones from the checkpoint,
        # forcing these arguments to be equal otherwise we can't even resume training.
        # The rest of the attributes (e.g. dropout) can stay as desired from command line, except for the block size.
        for arg in fixed_model_args:
            setattr(config, arg, checkpoint_model_args[arg])

        # Ensure consistency of fixed model args
        for arg in fixed_model_args:
            setattr(config, arg, checkpoint_model_args[arg])

        # Get block size
        ckpt_block_size = checkpoint_model_args["block_size"]
        required_block_size = config.block_size if config.block_size < ckpt_block_size else None
        setattr(config, "block_size", ckpt_block_size)

        # Create model instance with config and weights
        model = CustomGPTModel(config, checkpoint)

        # Eventually crop the model's block size
        if required_block_size is not None:
            print(f"Cropping model block size from {ckpt_block_size} to {required_block_size}")
            model.model.crop_block_size(required_block_size)
            setattr(config, "block_size", required_block_size)

        return model
    
     
    @classmethod
    def from_pretrained(cls, pretrained_model_path: str, safetensors=False, *model_args, **kwargs):
        print(f"Resuming training from {pretrained_model_path}")
        assert os.path.exists(
            pretrained_model_path
        ), f"Resuming from directory \'{pretrained_model_path}\' that does not exist!"

        # Load configuration from parameters or from the config file in the directory
        config = kwargs.pop('config', None)#, PretrainedConfig.from_pretrained(pretrained_model_path))
        if config is None:
            config = search_for_config(pretrained_model_path)

        # Update the config with the provided kwargs from parameters
        for arg in kwargs:
            if arg in fixed_model_args:
                print(f"Cannot change {arg} from the checkpoint, ignoring")
            else:
                setattr(config, arg, kwargs[arg])

        # Extract the model args from the checkpoint
        if safetensors:
            return cls.from_pretrained_safe_tensors(pretrained_model_path, config, *model_args, **kwargs)
        else:
            return cls.from_pretrained_pt(pretrained_model_path, config, *model_args, **kwargs)

    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        return self.model.generate(idx, max_new_tokens, temperature=temperature, top_k=top_k)
    
    def train(self, mode=True):
        super().train(mode)
        self.model.train(mode)


def master_model(model_name, n_checkpoint=None, *args, **kwargs):
    """
    Returns a CustomGPTModel object from the pretrained model path with the given configuration.
    """
    return CustomGPTModel.from_pretrained(model_name, n_checkpoint=n_checkpoint, *args, **kwargs)

def get_tokenizer():
    """
    Returns a Tokenizer object that implements the PreTrainedTokenizer interface.
    """
    tokenizer = ChessTokenizer(clean_up_tokenization_spaces=True)

    assert tokenizer.pad_token_id is not None

    return tokenizer
