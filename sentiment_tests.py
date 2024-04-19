import pandas as pd 
from nltk.corpus import stopwords
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
from numpy import argmax
from torch.nn.functional import softmax

model_name = "ProsusAI/finbert"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

df = pd.read_csv('FinalStockNews.csv')
for col in df.columns:
    col = col.strip
df.drop('Unnamed: 0', axis=1, inplace=True)
df.dropna(inplace=True)
df.reset_index(inplace=True, drop=True)

def sliding_window(text, tokenizer, max_length=512, overlap=50):

    tokenized_text = tokenizer.encode_plus(text, return_tensors='pt', add_special_tokens=True)
    input_ids = tokenized_text['input_ids'].squeeze().numpy().tolist()
    
    total_length = len(input_ids)
    step_size = max_length - overlap
    chunks = []
    
    for start in range(0, total_length, step_size):
        end = start + max_length
        chunk = input_ids[start:end]
        
        chunk = [tokenizer.cls_token_id] + chunk + [tokenizer.sep_token_id]
        chunks.append(chunk)
    
    return chunks

# unit test sliding window
assert sliding_window("This is the test", tokenizer, 2, 1) == [[101, 101, 2023, 102],
 [101, 2023, 2003, 102],
 [101, 2003, 1996, 102],
 [101, 1996, 3231, 102],
 [101, 3231, 102, 102],
 [101, 102, 102]]
assert sliding_window("The fox jumped over the lake and tackled Tom Brady", tokenizer, 6, 3) == [[101, 101, 1996, 4419, 5598, 2058, 1996, 102],
 [101, 5598, 2058, 1996, 2697, 1998, 26176, 102],
 [101, 2697, 1998, 26176, 3419, 10184, 102, 102],
 [101, 3419, 10184, 102, 102]]

def process_chunks(chunks, tokenizer, model):
    model.eval()  
    sentiments = []
    scores = []

    with torch.no_grad():
        for chunk in chunks:
            inputs = torch.tensor(chunk).unsqueeze(0)  
            

            inputs = torch.nn.functional.pad(inputs, (0, 512 - inputs.shape[1]), value=tokenizer.pad_token_id)
            
            outputs = model(inputs)
            logits = outputs.logits
            sentiment = torch.argmax(logits, dim=1).numpy()[0]  
            score = torch.softmax(logits, dim=1).max().item()  

            sentiments.append(sentiment)
            scores.append(score)

    return sentiments, scores

# unit tests for process_chunks
chunk1 = [[101, 101, 2023, 102],
 [101, 2023, 2003, 102],
 [101, 2003, 1996, 102],
 [101, 1996, 3231, 102],
 [101, 3231, 102, 102],
 [101, 102, 102]]
assert process_chunks(chunk1, tokenizer, model) == ([2, 2, 2, 2, 2, 2],
 [0.7957010269165039,
  0.8266777992248535,
  0.7992331385612488,
  0.7978023886680603,
  0.7663531303405762,
  0.7840284109115601])
chunk2 = [[101, 101, 1996, 4419, 5598, 2058, 1996, 102],
 [101, 5598, 2058, 1996, 2697, 1998, 26176, 102],
 [101, 2697, 1998, 26176, 3419, 10184, 102, 102],
 [101, 3419, 10184, 102, 102]]
assert process_chunks(chunk2, tokenizer, model) == ([2, 2, 2, 2],
 [0.7723026871681213,
  0.7744370698928833,
  0.7692530155181885,
  0.7506882548332214])

def aggregate_results(sentiments, scores):

    average_score_per_sentiment = {}
    for sentiment, score in zip(sentiments, scores):
        if sentiment not in average_score_per_sentiment:
            average_score_per_sentiment[sentiment] = []
        average_score_per_sentiment[sentiment].append(score)
    
    for sentiment in average_score_per_sentiment:
        average_score_per_sentiment[sentiment] = sum(average_score_per_sentiment[sentiment]) / len(average_score_per_sentiment[sentiment])
    
    dominant_sentiment = max(average_score_per_sentiment, key=average_score_per_sentiment.get)
    return dominant_sentiment, average_score_per_sentiment[dominant_sentiment]

# unit tests for aggregate_results
sent1 = [2, 2, 2, 2, 2, 2]
score1 = [0.7957010269165039,
  0.8266777992248535,
  0.7992331385612488,
  0.7978023886680603,
  0.7663531303405762,
  0.7840284109115601]
assert aggregate_results(sent1, score1) == (2, 0.7949659824371338)

sent2 = [2, 2, 2, 2, 2, 2]
score2 = [0.7957010269165039,
  0.8266777992248535,
  0.7992331385612488,
  0.7978023886680603,
  0.7663531303405762,
  0.7840284109115601]
assert aggregate_results(sent2, score2) == (2, 0.7949659824371338)

def full_window(text, tokenizer, model):
    chunks = sliding_window(text, tokenizer)
    sentiments, scores = process_chunks(chunks, tokenizer, model)
    return aggregate_results(sentiments, scores)


# unit test full_window
text1 = 'Contrary to popular belief, Lorem Ipsum is not simply random text. It has roots in a piece of classical Latin literature from 45 BC, making it over 2000 years old. Richard McClintock, a Latin professor at Hampden-Sydney College in Virginia, looked up one of the more obscure Latin words, consectetur, from a Lorem Ipsum passage, and going through the cites of the word in classical literature, discovered the undoubtable source. Lorem Ipsum comes from sections 1.10.32 and 1.10.33 of "de Finibus Bonorum et Malorum" (The Extremes of Good and Evil) by Cicero, written in 45 BC. This book is a treatise on the theory of ethics, very popular during the Renaissance. The first line of Lorem Ipsum, "Lorem ipsum dolor sit amet..", comes from a line in section 1.10.32.'
assert full_window(text1, tokenizer, model) == (2, 0.8298267126083374)

text2 = "It is a long established fact that a reader will be distracted by the readable content of a page when looking at its layout."
assert full_window(text2, tokenizer, model) == (2, 0.5958979725837708)

print("Success!")