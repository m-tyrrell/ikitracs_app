import numpy as np
import torch
import streamlit as st
from transformers import BertTokenizer
from transformers import BertForSequenceClassification

st.title('Test Classifier')
st.markdown('NDC Economy Wide')


from transformers import AutoTokenizer, AutoModelForSequenceClassification

# tokenizer = AutoTokenizer.from_pretrained("mtyrrell/ikitracs_economywide")
# model = AutoModelForSequenceClassification.from_pretrained("mtyrrell/ikitracs_economywide")



tokenizer = AutoTokenizer.from_pretrained("mgreenbe/bertlet-base-uncased-for-sequence-classification")
model = AutoModelForSequenceClassification.from_pretrained("mgreenbe/bertlet-base-uncased-for-sequence-classification")

text = st.text_area('Text Input')


def run_model(input_text):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_text = str(input_text)
    input_text = ' '.join(input_text.split())
    # input_tokenized = tokenizer.encode(input_text, return_tensors='pt').to(device)
    input_tokenized = tokenizer.encode(input_text, return_tensors='pt')
    pred = model(input_tokenized)                                          
    pred_int = np.argmax(pred['logits'].detach().cpu().numpy())
    output = ['NEGATIVE' if pred_int == 0 else 'ECONOMY-WIDE']

    st.write('Summary')
    st.success(output)

if st.button('Submit'):
    run_model(text)