import streamlit as st
import torch
import pickle
from models.classes import Seq2SeqTransformer, Encoder, Decoder, initialize_weights
from utils import get_text_transform, en_tokenizer, my_tokenizer
import sys
import os

# Print Python search paths
print("PYTHONPATH:", sys.path)

# Add the parent directory of "app" to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print("Updated PYTHONPATH:", sys.path)

# Check if the library path is accessible
library_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../library'))
print("Library Path Exists:", os.path.exists(library_path))

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Update token_transform
token_transform = {
    "en": en_tokenizer,
    "my": my_tokenizer
}
# Load metadata (vocabulary & tokenizer)
@st.cache_resource
def load_meta():
    meta_path = "models/meta-additive.pkl"
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    return meta

meta = load_meta()

# Define Transformers
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<sos>', '<eos>']

token_transform = meta['token_transform']
vocab_transform = meta['vocab_transform']
text_transform = get_text_transform(token_transform, vocab_transform)

SRC_LANGUAGE = "en"
TRG_LANGUAGE = "my"

# Model Configuration
input_dim = len(vocab_transform["en"])
output_dim = len(vocab_transform["my"])
hid_dim = 256
enc_layers = 3
dec_layers = 3
enc_heads = 8
dec_heads = 8
enc_pf_dim = 512
dec_pf_dim = 512
enc_dropout = 0.1
dec_dropout = 0.1

SRC_PAD_IDX = PAD_IDX
TRG_PAD_IDX = PAD_IDX

# Load the trained model
@st.cache_resource
def load_model():
    enc = Encoder(input_dim, hid_dim, enc_layers, enc_heads, enc_pf_dim, enc_dropout, device, max_length=1000)
    dec = Decoder(output_dim, hid_dim, dec_layers, dec_heads, dec_pf_dim, dec_dropout, device, mech="add", max_length=1000)
    model = Seq2SeqTransformer(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)
    model.load_state_dict(torch.load(r"C:\Users\st124\OneDrive\Desktop\NLP_A3\app\models\add_best_model.pt", map_location=device))
    model.eval()
    return model


model = load_model()
st.success("✅ Model Loaded Successfully!")

# Translation Function
def translate_text(sentence):
    model.eval()

    # Tokenize & numericalize input
    src_text = text_transform[SRC_LANGUAGE](sentence).to(device)
    src_text = src_text.unsqueeze(0)

    # Encode source sentence
    src_mask = model.make_src_mask(src_text)
    with torch.no_grad():
        enc_output = model.encoder(src_text, src_mask)

    # Greedy decoding
    input_tokens = [SOS_IDX]
    outputs = []
    max_seq = 100

    for _ in range(max_seq):
        with torch.no_grad():
            trg_tensor = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0).to(device)
            trg_mask = model.make_trg_mask(trg_tensor)
            output, _ = model.decoder(trg_tensor, enc_output, trg_mask, src_mask)

        pred_token = output.argmax(2)[:, -1].item()
        input_tokens.append(pred_token)
        outputs.append(pred_token)
        
        if pred_token == EOS_IDX:
            break

    # Convert token IDs to words
    trg_tokens = [vocab_transform[TRG_LANGUAGE].get_itos()[i] for i in outputs]

    # Format final translation
    translated_text = " ".join(trg_tokens[1:-1])
    return translated_text

# Streamlit App UI
st.title("English to Myanmar Translator")
st.subheader("Powered by Additive Attention Model")

# User Input
user_input = st.text_area("Enter an English sentence:", "")

if st.button("Translate"):
    if user_input:
        translation = translate_text(user_input)
        st.success(f"Translated Sentence: {translation}")
    else:
        st.warning("⚠️ Please enter a sentence before translating.")

