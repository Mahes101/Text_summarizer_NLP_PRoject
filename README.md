Text Summarization Project Using Transformers
Introduction
This project focuses on text summarization using transformers, a modern deep learning approach for generating concise and coherent summaries from longer text documents. We will use the transformers library alongside other essential libraries to achieve this.

Libraries and Imports
python
from transformers import Pipeline, set_seed
from datasets import load_dataset, load_from_disk
import matplotlib.pyplot as plt
import pandas as pd
from evaluate import load as load_metric
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import nltk
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import torch
Setup and Configuration
Load and Prepare Data:

python
# Load dataset
dataset = load_dataset('xsum')

# Display a sample
print(dataset['train'][0])
Load Pre-trained Model and Tokenizer:

python
# Load the model and tokenizer
model_name = "facebook/bart-large-cnn"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
Tokenize Data:

python
# Tokenize the dataset
def tokenize(batch):
    return tokenizer(batch['document'], truncation=True, padding='max_length')

tokenized_dataset = dataset.map(tokenize, batched=True)
Model Training
If fine-tuning is required, follow these steps:

Set Up Training Arguments:

python
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
)
Initialize Trainer:

python
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['validation'],
    tokenizer=tokenizer,
)
Train the Model:

python
trainer.train()
Model Evaluation
Generate Summaries:

python
def generate_summary(batch):
    inputs = tokenizer(batch["document"], return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    summary_ids = model.generate(inputs["input_ids"], num_beams=4, max_length=150, early_stopping=True)
    batch["summary"] = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return batch

summaries = tokenized_dataset["validation"].map(generate_summary, batched=True, batch_size=8)
Evaluate the Model:

python
metric = load_metric('rouge')

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions.argmax(-1)
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    return metric.compute(predictions=pred_str, references=label_str)

results = trainer.evaluate()
Conclusion
This documentation outlines the steps for setting up, training, and evaluating a text summarization model using transformers. The key components include data preparation, model loading, tokenization, training, and evaluation using the specified libraries.
