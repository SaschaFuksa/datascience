#%%
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords

#preprocessing steps:
# Tokenization
# Normalization
# POS Tagging
# Stemming or Lemmatization
# Stop word removal

#How to save:
# Into crawled csv as new columns?
# Create new CSV with ID and new preprocessed data? -> Arrays into Columns
# Create new CSV for each Location? -> location id as filename and each word gets a new column

def tokenization(text):
    #sentence tokenization - not sure if needed
    #sentence_tokens=sent_tokenize(text)
    #print("\nSentence Tokens: ",sentence_tokens)

    #word tokenization
    word_tokens=word_tokenize(text)
    print("\nWord Tokens: ",word_tokens)
    return word_tokens

def normalization(word_tokens):
    #remove punctuation
    word_tokens = [word for word in word_tokens if word.isalpha()]
    print("\nRemoved Puntuation: ",word_tokens)

    #normalization
    normed_word_tokens = [word.lower() for word in word_tokens]
    print("\nNormalized Word Tokens: ",normed_word_tokens)
    return normed_word_tokens

def posTagging(normed_word_tokens):
    pos_tags = nltk.pos_tag(normed_word_tokens)
    print("\nPos Tags: ",pos_tags)
    return pos_tags

def stemming(normed_word_tokens):
    ps = PorterStemmer()

    stemmed_words=[]
    for w in normed_word_tokens:
        stemmed_words.append(ps.stem(w))
    print("\nStemmed Words: ",stemmed_words)
    return stemmed_words

def lemmatization(pos_tags):    
    lem = WordNetLemmatizer()
    
    lemma_words = []
    for word in pos_tags:
        lemma_words.append(lem.lemmatize(word[0], get_wordnet_pos(word[1])))
    print("\nLemmatizated Words: ",lemma_words)
    return lemma_words

def stopWordRemoval(stemmed_words):
    stop_words=set(stopwords.words("english"))
    #Remove stopwords
    filtered_sent=[]
    stopwords_found=[]
    stemmed_words
    for w in stemmed_words:
        if w not in stop_words:
            filtered_sent.append(w)
        else:
            stopwords_found.append(w)
    print("\nFiltered Sentence:",filtered_sent)
    print("\nStopwords found:",stopwords_found)
    return filtered_sent

@staticmethod
def get_wordnet_pos(word_class_tag):
    """
    return WORDNET POS compliance to WORDENT lemmatization (a,n,r,v) 
    """
    if word_class_tag.startswith('J'):
        return wordnet.ADJ
    elif word_class_tag.startswith('V'):
        return wordnet.VERB
    elif word_class_tag.startswith('N'):
        return wordnet.NOUN
    elif word_class_tag.startswith('R'):
        return wordnet.ADV
    else:
        # As default pos in lemmatization is Noun
        return wordnet.NOUN



#%%
df_description = pd.read_csv('content/crawled_rough_guides.csv')
test_text = df_description.iloc[1,5]

#test preprocessing with one text
word_tokens = tokenization(test_text)
normed_word_tokens = normalization(word_tokens)
pos_tags = posTagging(normed_word_tokens)
stemmed_words = stemming(normed_word_tokens)
lemma_words = lemmatization(pos_tags)
final_words = stopWordRemoval(stemmed_words)
print("\nOriginal Text: ", test_text)
print("\nFinal Preprocessed Text: ", final_words)


# %%
# Downloads needed to run this code
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

