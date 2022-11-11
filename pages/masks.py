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

#Masking:
unmasker = pipeline('fill-mask', model='bert-base-uncased')

st.markdown("## **📌 Masked Text **")
with st.form(key="my_form"):

    st.markdown(" Masked token is [MASK]")
    doc = st.text_area(
        "Enter a text with a mask marked as [MASK]",
        height=100,
        value="Paris is the [MASK] of France."
    )
    text = doc

    submit_button = st.form_submit_button(label="Run")

if not submit_button:
    st.stop()


with st.spinner('Processing...'):
    mask_output = unmasker(text)
    mask_output_df = DataFrame(mask_output)
    st.markdown("## Text Unmasking ")
    st.header("")
    st.table(mask_output_df)




    st.markdown("## End")
    torch.cuda.empty_cache()
