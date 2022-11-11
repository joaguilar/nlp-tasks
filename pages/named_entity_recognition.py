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

#Named Entity Recognition
tokenizer_ner = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model_ner = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
# tokenizer_clinical = AutoTokenizer.from_pretrained("samrawal/bert-base-uncased_clinical-ner")
# model_clinical = AutoModelForTokenClassification.from_pretrained("samrawal/bert-base-uncased_clinical-ner")
tokenizer_clinical = AutoTokenizer.from_pretrained("d4data/biomedical-ner-all")
model_clinical = AutoModelForTokenClassification.from_pretrained("d4data/biomedical-ner-all")

model_ner.eval()
model_ner.to(device)
model_clinical.eval()
model_clinical.to(device)

st.markdown("## **📌 Named Entity Recognition **")
with st.form(key="my_form"):
    ce, c1, ce, c2, c3 = st.columns([0.07, 2, 0.07, 5, 0.07])
    with c1:
        ModelType = st.radio(
        "Choose the model to use",
        ["bert-base-NER", "biomedical-ner-all"],
        help="You can only choose between these two models."
        )
        if ModelType == "bert-base-NER":
            modelo=model_ner
            tokenizer = tokenizer_ner
        else:
            modelo=model_clinical
            tokenizer = tokenizer_clinical

    with c2:

        doc = st.text_area(
            "Enter a Text (English)",
            height=110,
            value="My name is Jose I am working with Danny in our San Jose office."
        )
        text = doc

    submit_button = st.form_submit_button(label="Run")

if not submit_button:
    st.stop()


with st.spinner('Processing...'):
    # ner = pipeline("ner", model=model_ner, tokenizer=tokenizer_ner,device=0)
    ner = pipeline("ner", model=modelo, tokenizer=tokenizer,device=0, aggregation_strategy="simple")
    ner_results = ner(text)
    print(ner_results)
    st.table(DataFrame(ner_results))

    if ModelType == "bert-base-NER":
        st.markdown("""
    |  Abbreviation | Description  |
    |---|---|
    | O  | Outside of a named entity  |
    | B-MIS  | Beginning of a miscellaneous entity right after another miscellaneous entity  |
    | I-MIS  | Miscellaneous entity  |
    | B-PER  | Beginning of a person’s name right after another person’s name  |
    | I-PER  | Person’s name  |
    | B-ORG  | Beginning of an organization right after another organization |
    | I-ORG  | Organization  |
    | B-LOC  | Beginning of a location right after another location  |
    | I-LOC  | Location  |
        """)

    st.markdown("## End")
    torch.cuda.empty_cache()
