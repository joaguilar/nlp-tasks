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


print("Init....")

#Constants:
TEXT_GENERATION = 1
ENTITY_RECOGNITION = 2
SENTIMENT_ANALYSIS = 3
TEXT_MASK = 4

device = 'cpu'
if (torch.backends.mps.is_available()):
    # print('MPS: ' + str(torch.backends.mps.is_available()))
    device = 'mps'
    device = 'cpu' # Problems whe inferencing with current version of PyTorch

if (torch.cuda.is_available()):
    # print('Using CUDA: ' + str(torch.cuda.is_available()))
    device = 'cuda'
print('Using device: '+device)


st.session_state['device'] = device

# Streamlit
st.set_page_config(
    page_title="Exmples of NLP tasks",
    page_icon="",
)


def _max_width_():
    max_width_str = f"max-width: 1400px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )

_max_width_()

st.write("""
# NLP Tasks

## Select a task from the left panel
""")