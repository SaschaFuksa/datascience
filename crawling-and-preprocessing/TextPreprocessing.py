#%%
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords

#Preprocessing Ablauf:
# Tokenization
# Normalization
# POS Tagging
# Stemming or Lemmatization
# Stop word removal

def sentence_tokenization(text):
    sentence_token=sent_tokenize(text)
    #print("\nSentence Tokens: ",sentence_tokens)
    return sentence_token

def tokenization(text):
    #word tokenization
    word_tokens=word_tokenize(text)
    #print("\nWord Tokens: ",word_tokens)
    return word_tokens

def removePunctuation(word_tokens):
    #remove punctuation
    word_tokens = [word for word in word_tokens if word.isalpha()]
    #print("\nRemoved Puntuation: ",word_tokens)
    return word_tokens

def normalization(word_tokens):
    #normalization
    normed_word_tokens = [word.lower() for word in word_tokens]
    #print("\nNormalized Word Tokens: ",normed_word_tokens)
    return normed_word_tokens

def posTagging(normed_word_tokens):
    pos_tags = nltk.pos_tag(normed_word_tokens)
    #print("\nPos Tags: ",pos_tags)
    return pos_tags

def stemming(normed_word_tokens):
    ps = PorterStemmer()

    stemmed_words=[]
    for w in normed_word_tokens:
        stemmed_words.append(ps.stem(w))
    #print("\nStemmed Words: ",stemmed_words)
    return stemmed_words

def lemmatization(pos_tags):    
    lem = WordNetLemmatizer()
    
    lemma_words = []
    for word in pos_tags:
        lemma_words.append(lem.lemmatize(word[0], get_wordnet_pos(word[1])))
    #print("\nLemmatizated Words: ",lemma_words)
    return lemma_words

def stopWordRemoval(stemmed_words):
    stop_words=set(stopwords.words("english"))
    #Remove stopwords
    filtered_sent=[]
    stopwords_found=[]
    for w in stemmed_words:
        if w not in stop_words:
            filtered_sent.append(w)
        else:
            stopwords_found.append(w)
    #print("\nFiltered Sentence:",filtered_sent)
    #print("\nStopwords found:",stopwords_found)
    return filtered_sent

def stopWordRemovalDF(df_tokens):
    stop_words = set(stopwords.words("english"))
    
    for row in df_tokens.itertuples():
        if row.stemmed_token in stop_words:
            #print('Remove: ', row.stemmed_token)
            df_tokens.drop(row.Index, inplace=True)
    return df_tokens

def preProcessing(label, text):
    word_tokens = tokenization(text)
    word_tokens = removePunctuation(word_tokens)
    normed_word_tokens = normalization(word_tokens)
    pos_tags = posTagging(normed_word_tokens)
    stemmed_words = stemming(normed_word_tokens)
    lemma_words = lemmatization(pos_tags)
    final_words = stopWordRemoval(stemmed_words)

    df_tokens = pd.DataFrame({'token':word_tokens, 'normed_token':normed_word_tokens, 'pos_tags':pos_tags, 'stemmed_token':stemmed_words, 'lemma_token':lemma_words})
    df_tokens = stopWordRemovalDF(df_tokens)
    df_tokens['class'] = label

    return df_tokens

def get_wordnet_pos(word_class_tag):
    if word_class_tag.startswith('J'):
        return wordnet.ADJ
    elif word_class_tag.startswith('V'):
        return wordnet.VERB
    elif word_class_tag.startswith('N'):
        return wordnet.NOUN
    elif word_class_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


#%%

def main():
    df_places = pd.read_csv('content/crawled_rough_guides.csv')

    for row in df_places.itertuples():
        df_introduction = preProcessing('introduction', row.introduction)
        df_description = preProcessing('description', row.description)

        file_name = str(row.number) + "_" + row.place.replace(" ", "_") + '.csv'
        file_path = 'content/processed_places/'

        df_export = pd.concat([df_introduction, df_description], ignore_index=True)
        df_export.to_csv(file_path+file_name, index=False)

        print("created file: " + file_path + file_name)
    
# %%
# Downloads needed to run this python-code
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('stopwords')

