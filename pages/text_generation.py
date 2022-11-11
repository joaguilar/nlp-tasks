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

def cut_last_incomplete_sentence(text:str):
    return text.rsplit('.',1)[0] + '.'


device = st.session_state['device']

# Text Generation
tokenizer_distill = AutoTokenizer.from_pretrained("distilgpt2")

model_distill = AutoModelForCausalLM.from_pretrained("distilgpt2")

tokenizer_gpt2 = AutoTokenizer.from_pretrained("gpt2")

model_gpt2 = AutoModelForCausalLM.from_pretrained("gpt2")

model_distill.eval()
model_distill.to(device)
model_gpt2.eval()
model_gpt2.to(device)

top_k = 50

st.markdown("""
# NLP Tasks

## Text Generation
""")

with st.form(key="my_form"):
    ce, c1, ce, c2, c3 = st.columns([0.07, 2, 0.07, 5, 0.07])
    with c1:
        ModelType = st.radio(
        "Choose the model to use",
        ["DistilGPT2", "GPT-2"],
        help="You can only choose between these two models."
        )
        if ModelType == "DistilGPT2":
            modelo=model_distill
            tokenizer = tokenizer_distill
        else:
            modelo=model_gpt2
            tokenizer = tokenizer_gpt2
        
        no_textos_generar = st.slider(
            "# Text to generate",
            min_value=1,
            max_value=5,
            value=3,
            help="Number of texts to generate.",
        )
        tam_texto_generado = st.number_input(
            "Max Size (in chars) of generated text",
            min_value=200,
            max_value=400,
            help="Min/Max size of generated text."
        )
        
    with c2:
        doc = st.text_area(
            "Type a couple of sentences as a prompt. (max 500 palabras)",
            height=310,
            value="In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English."
        )
        MAX_WORDS = 200
        res = len(re.findall(r"\w+", doc))
        if res > MAX_WORDS:
            st.warning(
                "⚠️ Your text has "
                + str(res)
                + " words."
                + " Only the first 200 words will be considered."
            )

        text = doc[:MAX_WORDS]

    submit_button = st.form_submit_button(label="Run")

if not submit_button:
    st.stop()

print("Text: "+text)

with st.spinner('Processing...'):
    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
    max_length = input_ids.shape[1]
    flat_input_ids = torch.flatten(input_ids,start_dim = 1)
    textos = modelo.generate(input_ids, pad_token_id=50256,
                                    do_sample=True, 
                                    max_length=tam_texto_generado, 
                                    min_length=200,
                                    top_k=50,
                                    num_return_sequences=no_textos_generar)
    mostrar = []

    for i, sample_output in enumerate(textos):
        # salida_texto_temp = tokenizer_diputados.decode(sample_output.tolist())
        salida_texto_temp = tokenizer.decode(sample_output.tolist())
        salida_texto = cut_last_incomplete_sentence(salida_texto_temp)
        mostrar.append(
            {
                "Generated Texts":salida_texto
            }
        )
    #     print(">> Generated text {}\n\n{}".format(i+1, salida_texto))
    #     # seq = random.randint(0,100000)
    # #     with open('/content/textos/ejemplo_diputado_'+str(seq)+'.txt','w') as f:
    # #       f.write(salida_texto)
    #     print('\n---')

    df = (
        DataFrame(mostrar, columns=["Generated Texts"])
        .reset_index(drop=True)
    )

    # Add styling
    cmGreen = sns.light_palette("green", as_cmap=True)
    cmRed = sns.light_palette("red", as_cmap=True)
    # df = df.style.background_gradient(
    #     cmap=cmGreen
    # )
    # print("Dataframe:")
    # print(df.to_string())
    # print("Dataframe/")

    st.markdown("## Generated Texts")

    st.header("")

    st.table(df.assign(hack='').set_index('hack'))
    torch.cuda.empty_cache()

    st.markdown("## End")
