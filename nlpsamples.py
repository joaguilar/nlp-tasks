import torch
import random
import streamlit as st
from pandas import DataFrame
import seaborn as sns

from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

from transformers import logging
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
from tokenizers import ByteLevelBPETokenizer
logging.set_verbosity_warning()


print("Init....")

device = 'cpu'
if (torch.backends.mps.is_available()):
    # print('MPS: ' + str(torch.backends.mps.is_available()))
    device = 'mps'
    device = 'cpu' # Problems whe inferencing with current version of PyTorch

if (torch.cuda.is_available()):
    # print('Using CUDA: ' + str(torch.cuda.is_available()))
    device = 'cuda'
print('Using device: '+device)

tokenizer_distill = AutoTokenizer.from_pretrained("distilgpt2")

model_distill = AutoModelForCausalLM.from_pretrained("distilgpt2")

tokenizer_gpt2 = AutoTokenizer.from_pretrained("gpt2")

model_gpt2 = AutoModelForCausalLM.from_pretrained("gpt2")

model_gpt2.eval()
model_gpt2.to(device)

top_k = 50


# Streamlit
st.set_page_config(
    page_title="DPT-2: Modelo de Lenguaje GPT-2 aplicado la generación de texto de discursos políticos.",
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

## Text Generation
""")

st.markdown("## **📌 Enter a prompt (begining of a text...) **")
with st.form(key="my_form"):
    ce, c1, ce, c2, c3 = st.columns([0.07, 2, 0.07, 5, 0.07])
    with c1:
        ModelType = st.radio(
           "Escoja el Modelo a Utilizar",
           ["DistilGPT2", "GPT-2-small-spanish"],
           help="Solamente es posible escojer entre estos dos modelos."
        )
        if ModelType == "DPT-2":
           modelo=model_diputados
           tokenizer = tokenizer_diputados
        else:
           modelo=model_gpt2
           tokenizer = tokenizer_gpt2
        
        no_textos_generar = st.slider(
            "# de textos a generar",
            min_value=1,
            max_value=5,
            value=3,
            help="Textos a generar, entre 1 y 5, por defecto 3.",
        )
        tam_texto_generado = st.number_input(
            "Tamaño del texto a generar",
            min_value=200,
            max_value=400,
            help="Tamaños mínimos y máximos del texto a generar."
        )
        
    with c2:
        doc = st.text_area(
            "Digite las primeras dos oraciones de un discurso y observe como el modelo le genera el resto del discurso. (max 500 palabras)",
            height=310
        )
        MAX_WORDS = 200
        import re
        res = len(re.findall(r"\w+", doc))
        if res > MAX_WORDS:
            st.warning(
                "⚠️ Su texto contiene "
                + str(res)
                + " palabras."
                + " Solamente las primeras 200 palabras serán utilizadas."
            )

        text = doc[:MAX_WORDS]

    submit_button = st.form_submit_button(label="Generar Textos")

if not submit_button:
    st.stop()