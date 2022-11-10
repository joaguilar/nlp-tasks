# nlp-tasks
Examples of NLP Tasks using Huggingface models and Streamlit web framework

# Installation

1. Create a conda environment:

```
conda create --prefix .\nlp-tasks python=3.9
```

2. Activate the environment:

```
conda activate .\nlp-tasks
```

3. Install prerequisites:

Install Pytorch with CUDA acceleration (if possible). Follow the commands from: https://pytorch.org/

Command used for my installation:

```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```

Then install the remaining requirements:

```
pip install -r requirements.txt
```

4. Run The Streamlit webapp

```
streamlit run nlpsamples.py
```

