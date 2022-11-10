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

tokenizer_distill = AutoTokenizer.from_pretrained("distilgpt2")

model_distill = AutoModelForCausalLM.from_pretrained("distilgpt2")

tokenizer_gpt2 = AutoTokenizer.from_pretrained("gpt2")

model_gpt2 = AutoModelForCausalLM.from_pretrained("gpt2")

model_gpt2.eval()
model_gpt2.to(device)

top_k = 50

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

#Masking:
unmasker = pipeline('fill-mask', model='bert-base-uncased')

#Named Entity Recognition
tokenizer_ner = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model_ner = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
model_ner.eval()
model_ner.to(device)



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

## Text Generation
""")
selected = TEXT_GENERATION
st.markdown("## **📌 Enter a prompt (begining of a text...) **")
with st.form(key="my_form"):
    tab1, tab2, tab3, tab4 = st.tabs(["Text Generation", "Entity Recognition", "Sentiment Analysis", "Masks"])
    with tab1:
        ce, c1, ce, c2, c3 = st.columns([0.07, 2, 0.07, 5, 0.07])
        with c1:
            selected = TEXT_GENERATION
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
            res = len(re.findall(r"\w+", doc))
            if res > MAX_WORDS:
                st.warning(
                    "⚠️ Su texto contiene "
                    + str(res)
                    + " palabras."
                    + " Solamente las primeras 200 palabras serán utilizadas."
                )

            text = doc[:MAX_WORDS]
        with tab2:
            print("Tab 2")
            st.markdown("## **📌 Named Entity Recognition **")
            doc = st.text_area(
                "Enter a Text (English)",
                height=110,
                value="My name is Jose I am working with Danny in our San Jose office."
            )
            text = doc
            selected = ENTITY_RECOGNITION
            print("Text "+text)

        with tab3:
            st.markdown("## **📌 Tweet Sentient Analysis **")
            doc = st.text_area(
                "Enter a Tweet (English)",
                height=110,
                value="Good morning, I love everybody!!"
            )
            text = doc
            selected = SENTIMENT_ANALYSIS


        with tab4:
            st.markdown("## **📌 Masked Text **")
            st.markdown(" Masked token is [MASK]")
            doc = st.text_area(
                "Enter a text with a mask marked as [MASK]",
                height=100,
                value="Paris is the [MASK] of France."
            )
            text = doc
            selected = TEXT_MASK

    submit_button = st.form_submit_button(label="Run")

if not submit_button:
    st.stop()


with st.spinner('Processing...'):
    if selected == SENTIMENT_ANALYSIS:
        sentiment_output = []
        inputs = tokenizer(text, return_tensors="pt").to(device)
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
        # .assign(hack='').set_index('hack'))

        # predicted_class_id = logits.argmax().item()
        # label = model_sentiment.config.id2label[predicted_class_id]
        # print("Label = "+label)
    if selected == TEXT_MASK:
        mask_output = unmasker(text)
        mask_output_df = DataFrame(mask_output)
        st.markdown("## Text Unmasking ")
        st.header("")
        st.table(mask_output_df)

    if selected == ENTITY_RECOGNITION:
        ner = pipeline("ner", model=model_ner, tokenizer=tokenizer)
        ner_results = ner(text)
        print(ner_results)



    st.markdown("## End")
