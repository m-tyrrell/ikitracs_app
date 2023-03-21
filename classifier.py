import pandas as pd
import numpy as np
import os
import random


import torch
import streamlit as st
from transformers import BertTokenizer
from transformers import BertForSequenceClassification

st.title('Test Classifier')
st.markdown('NDC Economy Wide')

# model = st.selectbox('Select the model', ('BART', 'T5'))

# if model == 'BART':
#     _num_beams = 4
#     _no_repeat_ngram_size = 3
#     _length_penalty = 1
#     _min_length = 12
#     _max_length = 128
#     _early_stopping = True
# else:
#     _num_beams = 4
#     _no_repeat_ngram_size = 3
#     _length_penalty = 2
#     _min_length = 30
#     _max_length = 200
#     _early_stopping = True

model = BertForSequenceClassification.from_pretrained(os.path.join('ikitracs_economywide'), num_labels=2)
tokenizer = BertTokenizer.from_pretrained(os.path.join('ikitracs_economywide'), num_labels=2)

# col1, col2, col3 = st.beta_columns(3)
# _num_beams = col1.number_input("num_beams", value=_num_beams)
# _no_repeat_ngram_size = col2.number_input("no_repeat_ngram_size", value=_no_repeat_ngram_size)
# _length_penalty = col3.number_input("length_penalty", value=_length_penalty)

# col1, col2, col3 = st.beta_columns(3)
# _min_length = col1.number_input("min_length", value=_min_length)
# _max_length = col2.number_input("max_length", value=_max_length)
# _early_stopping = col3.number_input("early_stopping", value=_early_stopping)

text = st.text_area('Text Input')


def run_model(input_text):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_text = str(input_text)
    input_text = ' '.join(input_text.split())
    input_tokenized = tokenizer.encode(input_text, return_tensors='pt').to(device)
    pred = model(input_tokenized)                                          
    pred_int = np.argmax(pred['logits'].detach().cpu().numpy())
    output = ['NEGATIVE' if pred_int == 0 else 'ECONOMY-WIDE']

    st.write('Summary')
    st.success(output)

if st.button('Submit'):
    run_model(text)