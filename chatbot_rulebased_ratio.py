import streamlit as st
import numpy as np 
import string
from nltk.corpus import stopwords
import pandas as pd 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from Levenshtein import ratio
from nltk.stem import PorterStemmer
pd.set_option('display.max_colwidth', None)
from textblob import Word
import random
import re

import warnings
warnings.filterwarnings('ignore')

import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
st.title(' "SARTHI" THE CHATBOT FOR YOUR QUESTIONS : ')
st.subheader('Please ask question related to Supervised Learning ')

human_text = st.text_input("")

data=pd.read_csv('final_data.csv',encoding = 'unicode_escape',)
data.reset_index(drop=True,inplace=True)

data['question_clean'] = data['Questions'].str.replace('[^\w\s]','')
data['question_clean'] = data['question_clean'].str.replace('-',' ')
data['question_clean'] = data['question_clean'].str.replace('[?“”]','')
data['question_clean'] = data['question_clean'].apply(word_tokenize)
data['question_clean'] = data['question_clean'].apply(lambda x: [word.lower() for word in x])
stop_words = stopwords.words('english')
new_words=('explain','following','given','describe','ie','find','using','suppose','define','meaning','image','name','required','interview','discus''detail','indepth','short','brief','number','12','many','give','help','used','one','two','could','Algorithm','algorithm','basic','difference','respect','model','choose','optimal', 'value','ÃÂÃ','ÃÃÂÃ','Â','ÂÂ','use','used','ââââ','â','ââ','âââ')
for i in new_words:
    stop_words.append(i)
data['question_clean'] = data['question_clean'].apply(lambda x: [word for word in x if word not in stop_words])
data['question_clean'] = [' '.join(map(str, l)) for l in data['question_clean']]
stemmer=PorterStemmer()
data['question_clean'] = data['question_clean'].apply(lambda x: " ".join(sorted([stemmer.stem(word) for word in x.split()])))
wordnet_lemmatizer = WordNetLemmatizer()
data['question_clean'] = data['question_clean'].apply(lambda x: " ".join(sorted([wordnet_lemmatizer.lemmatize(word) for word in x.split()])))

#Greetings
greeting_inputs = ("hey", "good morning", "good evening", "morning", "evening", "hi", "whatsup","hello")
greeting_responses = ["hey", "hey hows you?", "hello, how you doing", "hello", "Welcome, I am good and you"]
def generate_greeting_response(greeting):
    for token in greeting.split():
        if token.lower() in greeting_inputs:
            return random.choice(greeting_responses)

def generate_response(user_input,data,stop_words):
    sarthi_response = ''
    user_input=str(user_input)
    user_input =re.sub(r'[^\w\s]', '', user_input)
    user_input = user_input.replace('-',' ')
    user_input = re.sub('[\?\.\/]',' ',user_input)
    user_input = user_input.split()
    user_input = (lambda user_input: [word.lower() for word in user_input])(user_input)
    user_input = (lambda user_input: [word for word in user_input if word not in stop_words])(user_input)
    stemmer=PorterStemmer()
    user_input = (lambda user_input:sorted([stemmer.stem(word) for word in user_input]))(user_input)
    wordnet_lemmatizer = WordNetLemmatizer()
    user_input = (lambda user_input:" ".join(sorted([wordnet_lemmatizer.lemmatize(word) for word in user_input])))(user_input)
    
    match_count=data["question_clean"].str.contains(user_input).sum()
    if match_count==0:
        sarthi_response = sarthi_response + "So sorry, I am designed to discuss on supervised machine learning only."
        return sarthi_response
    else:
        for idx, row in data.iterrows():
            score = ratio(row["question_clean"], user_input)
            if score >= 0.85: # I'm sure, stop here
                return row["Answer"]
        else:
            f_ans=sarthi_response + "I am sorry, I could not understand you."
        return f_ans       
        
continue_dialogue = True
st.write("Hello, I am your friend SARTHI. You can ask me any question regarding supervised learning:")
if human_text:
    st.write("\nUSER:",end=" ")   
    human_txt = human_text.lower()
    if human_txt != 'bye':
        if human_txt == 'thanks' or human_txt == 'thank you very much' or human_txt == 'thank you':
            continue_dialogue = False
            st.write("SARTHI: Most welcome")
        else:
            if generate_greeting_response(human_txt) != None:
                st.write("\nSARTHI: " + generate_greeting_response(human_txt))
            else:              
                st.write("\nSARTHI: ", end="")
                st.write(generate_response(human_txt,data,stop_words))
    else:
        continue_dialogue = False
        st.write("SARTHI: Good bye and take care of yourself...")