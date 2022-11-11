import torch
import random
import streamlit as st
from pandas import DataFrame
import seaborn as sns
import re
import urllib.request
import csv
import numpy as np
import extra_streamlit_components as stx
from scipy.special import softmax

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, pipeline, AutoModelForTokenClassification

from transformers import logging
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
from tokenizers import ByteLevelBPETokenizer

logging.set_verbosity_warning()
device = st.session_state['device']

#Sentiment Analysis:
# download label mapping
labels=[]
task = 'sentiment'
mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
with urllib.request.urlopen(mapping_link) as f:
    html = f.read().decode('utf-8').split("\n")
    csvreader = csv.reader(html, delimiter='\t')
labels = [row[1] for row in csvreader if len(row) > 1]


tokenizer_sentiment = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

model_sentiment = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

model_sentiment.eval()
model_sentiment.to(device)

st.markdown("## **📌 Tweet Sentiment Analysis **")

with st.form(key="my_form"):
    doc = st.text_area(
        "Enter a Tweet (English)",
        height=110,
        value="Good morning, I love everybody!!"
    )
    text = doc

    submit_button = st.form_submit_button(label="Run")

if not submit_button:
    st.stop()


with st.spinner('Processing...'):
    sentiment_output = []
    inputs = tokenizer_sentiment(text, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model_sentiment(**inputs).logits
    output = model_sentiment(**inputs)
    scores = output[0][0].detach().cpu().numpy()
    scores = softmax(scores)
    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    for i in range(scores.shape[0]):
        l = labels[ranking[i]]
        s = scores[ranking[i]]
        # sentiment_output[i]={
        #     l:str(np.round(float(s), 4))
        # }
        sentiment_output.append(
            {
                    "Sentiment":l,
                    "Prediction":str(np.round(float(s), 4))
            }
        ) 
        # print(f"{i+1}) {l} {np.round(float(s), 4)}")
    print(sentiment_output)
    # sentiment_output_df = DataFrame(sentiment_output,columns=['Sentiment','value'])
    # sentiment_output_df = DataFrame.from_dict(sentiment_output,orient='index', columns=["Sentiment","Prediction"])
    sentiment_output_df = DataFrame(sentiment_output)
    print(sentiment_output_df)
    st.markdown("## Text Sentiment")

    st.header("")

    st.table(sentiment_output_df)

    torch.cuda.empty_cache()



    st.markdown("## End")