from genericpath import exists
import os
import spacy
import numpy as np
from tqdm import tqdm
from transformers import pipeline

data_types = ['text', 'video', 'audio', 'speech', 'image']

languages = [
    'English', 'Chinese', 'Spanish', 'Hindi', 'Bengali', 'Portuguese', 'Russian', 
    'Japanese', 'Vietnamese', 'German', 'French', 'Turkish', 'Korean', 'Italian',
    'Polish', 'Dutch', 'Indonesian', 'Thai', 'Danish', 'Czech', 'Finnish', 'Greek',
    'Swedish', 'Hungarian', 'Latvian', 'Lithuanian', 'Estonian', 'Arabic', 'Multilingual'
]

act_functions = [
    'relu', 'silu', 'gelu', 'sigmoid', 'tanh', 'elu', 'softmax',
]

architectures = [
    {'CNN': 'cnn'}, 
    {'DNN': ['dnn', 'ann']}, # should be treated together
    {'RNN' : ['rnn']}, 
    {'LSTM' : ['lstm']}, 
    {'GRU' : ['gru']}, 
    {'GAN' : ['gan']}, 
    {'VAE' : ['vae']}, 
    {'seq2seq' : ['seq2seq']}, 
    {'BERT': ['bert']}, 
    {'Transformer' : ['transformer']},
    {'GPT' : ['gpt']}, 
    {'GPT-2': ['gpt2', 'gpt-2']}, # should be treated together
    {'GPT-3': ['gpt3', 'gpt-3']}, # should be treated together
    {'AE': ['ae', 'autoencoder']}, # should be treated together
    {'ResNet': ['resnet']},
    {'attention': ['attention']},
    {'NER': ['ner']},
    {'ViT': ['vit']}
]

arch_list = [list(arch.keys())[0] for arch in architectures]

topics = [
    'Art',
    'Biology',
    'Business',
    'Chemistry',
    'Computer Science',
    'Economics',
    'Engineering',
    'Environmental Science',
    'Geography',
    'Geology',
    'History',
    'Materials Science',
    'Mathematics',
    'Medicine',
    'Philosophy',
    'Physics',
    'Political Science',
    'Psychology',
    'Sociology'
]

nlp = spacy.load("en_core_web_lg")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0)

quick = np.load('../data/additional_info_dict.npy', allow_pickle=True).item()
keys = list(quick.keys())

def concetenate_texts(q):
    texts = []
    for key in keys:
        d = q[key]
        text = ''
        if d['title'] is not None:
            text += d['title'] + '\n'
        if d['abstract'] is not None:
            text += d['abstract'] + '\n'
        if d['tldr'] is not None:
            text += d['tldr']['text']
        
        q[key]['text'] = text
        texts.append(text)
    return q, texts

def predict_datatype(q, texts):
    data_type_preds = classifier(texts, data_types)
    data_types_preds_words = list(map(lambda x: x['labels'][0], data_type_preds))

    for key, dt in tqdm(zip(keys, data_types_preds_words)):
        q[key]['data_type'] = [dt]
    return q

def apply_spacy(q):
    for key in tqdm(keys, desc='Adding values to categories'):
        q[key]['language'] = []
        q[key]['act_function'] = []
        q[key]['architecture'] = []
        doc = nlp(q[key]['text'])
        
        for token in doc:
            if token.text in languages:
                q[key]['language'].append(token.text)
            if token.text.lower() in act_functions:
                q[key]['act_function'].append(token.text)
            for arch in architectures:
                arch_key = list(arch.keys())[0]
                if token.text.lower() in arch[arch_key]:
                    q[key]['architecture'].append(arch_key)

    for key in tqdm(keys, desc='Making cat. values unique and concetenate authors'):
        for name in ['language', 'act_function', 'architecture']:
            q[key][name] = list(set(q[key][name]))

        if q[key]['authors'] is not None:
            q[key]['authors_string'] = ', '.join(list(map(lambda x: x['name'], q[key]['authors'])))
    return q


def main():
    quick, texts = concetenate_texts(quick)
    quick = predict_datatype(quick, texts)
    quick = apply_spacy(quick)

    np.save('../data/full_data_dict.npy', quick)

    os.makedir('../data/lists/', exists_ok=True)
    np.save('../data/lists/topics.npy', np.array(topics))
    np.save('../data/lists/data_types.npy', np.array(data_types))
    np.save('../data/lists/act_functions', np.array(act_functions))
    np.save('../data/lists/languages.npy', np.array(languages))
    np.save('../data/lists/archs.npy', np.array(arch_list))
