import torch
from transformers import MBartTokenizer

# Define source and target languages
SRC_LANGUAGE = 'en'
TRG_LANGUAGE = 'my'

# Define special symbols and indices
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
# Ensure special tokens are in order for proper insertion into vocab
special_symbols = ['<unk>', '<pad>', '<sos>', '<eos>']

# Initialize the MBart tokenizer
tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-50")

# Helper function to sequentially apply transformations
def sequential_transforms(*transforms):
    """
    Combines multiple transformations into a single function.
    
    Args:
        *transforms: Variable number of transformation functions.
        
    Returns:
        func: A function that applies all transformations in sequence.
    """
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# Function to add BOS/EOS tokens and convert to tensor
def tensor_transform(token_ids):
    """
    Adds <sos> and <eos> tokens to a sequence and converts to tensor.
    
    Args:
        token_ids (list): List of token indices.
        
    Returns:
        torch.Tensor: Tensor with <sos> and <eos> tokens added.
    """
    return torch.cat((torch.tensor([SOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))

# Function to tokenize using MBartTokenizer
def mbart_tokenize(text, lang):
    """
    Tokenizes text using MBartTokenizer and sets the language code.
    
    Args:
        text (str): Input text to be tokenized.
        lang (str): Language code ('en_XX' for English, 'my_MM' for Myanmar).
        
    Returns:
        list: Tokenized text as a list of token IDs.
    """
    tokenizer.src_lang = lang
    return tokenizer.encode(text, add_special_tokens=False)

# Function to get text transformations for both source and target languages
def get_text_transform(token_transform, vocab_transform):

    text_transform = {}
    for ln in [SRC_LANGUAGE, TRG_LANGUAGE]:
        text_transform[ln] = sequential_transforms(token_transform[ln],  # Tokenization
                                                   vocab_transform[ln],  # Numericalization
                                                   tensor_transform)     # Add BOS/EOS and convert to tensor
    return text_transform

from transformers import MBartTokenizer

# Initialize the MBart tokenizer
tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-50")

# Define tokenizer functions
def en_tokenizer(text):
    return tokenizer.tokenize(text)

def my_tokenizer(text):
    return tokenizer.tokenize(text)
