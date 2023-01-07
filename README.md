---
language: en
tags:
- exbert
license: apache-2.0
datasets:
- bookcorpus
- wikipedia
---

# BERT base model (uncased)

## Model description

Pretrained model on English language using a masked language modeling (MLM) objective. It was introduced in
[this paper](https://arxiv.org/abs/1810.04805) and first released in
[this repository](https://github.com/google-research/bert). This model is uncased: it does not make a difference
between english and English.

## Original implementation

Follow [this link](https://huggingface.co/bert-base-uncased) to see the original implementation.

## How to use

Download the model by cloning the repository via `git clone https://huggingface.co/OWG/bert-base-uncased`.

Then you can use the model with the following code:

```python
from onnxruntime import InferenceSession, SessionOptions, GraphOptimizationLevel
from transformers import BertTokenizer


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

options = SessionOptions()
options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

session = InferenceSession("path/to/model.onnx", sess_options=options)
session.disable_fallback()

text = "Replace me by any text you want to encode."
input_ids = tokenizer(text, return_tensors="pt", return_attention_mask=True)

inputs = {k: v.cpu().detach().numpy() for k, v in input_ids.items()}
outputs_name = session.get_outputs()[0].name

outputs = session.run(output_names=[outputs_name], input_feed=inputs)
```

This is a sentiment analysis model based on positive and negative reviews of the IMDB data set provided in Kaggle in which we utilise Bert pretrained model to train the neural network and achieve an accuracy of 86% on 5 epochs the accuracy can frther be tuned by increasing the number of epochs or by tuning the neural network activation functions. 

PS. Feel free to tune with the model and provide better results. 

Before using the bert based uncased mode, make sure you go through the hugging face documentation and download the required .bin,.onnx files that are the model files and download the vocab file by cloning the bert base uncased repo.

The dataset in csv format can also be downloaded from kaggle which contains 50,000 reviews approximately, the l;ink to the competition is 
"https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews".

This project was undertaken to better understand bert model.
