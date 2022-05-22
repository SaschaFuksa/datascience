#%%
import nltk
import pandas as pd

#read location token csv
df_location = pd.read_csv('content/processed_places/71_Freiburg.csv')
df_location

#%%
#read pos-tags into list
pos_tags = df_location['pos_tags']
pos_tags

#%%
#ne_chunk adds NE class to the list of token,POS tuples
chunk = nltk.ne_chunk(pos_tags)
print(chunk)

#%%
#extracts Named Entities from the list created before
NE = [ " ".join(w for w, t in ele) for ele in chunk if isinstance(ele, nltk.Tree)]
print (NE) 


#%%
import spacy 
import en_core_web_sm
nlp = en_core_web_sm.load()

#%%
# Write a function to display basic entity info: 
def show_ents(doc): 
    if doc.ents: 
        for ent in doc.ents: 
            print(ent.text+' - ' +str(ent.start_char) +' - '+ str(ent.end_char) +' - '+ent.label_+ ' - '+str(spacy.explain(ent.label_))) 
    else: 
        print('No named entities found.')

#%%        
doc1 = " ".join(df_location['token'])
doc1 = nlp(doc1)
show_ents(doc1)

# %%
# document level 
for e in doc1.ents: 
    print(e.text, e.start_char, e.end_char, e.label_) 
# OR 
ents = [(e.text, e.start_char, e.end_char, e.label_) for e in doc1.ents] 
print(ents)

#token level 
# doc[0], doc[1] ...will have tokens stored. 

ent_one = [doc1[3].text, doc1[3].ent_iob_, doc1[3].ent_type_] 
ent_two = [doc1[5].text, doc1[5].ent_iob_, doc1[5].ent_type_] 
print(ent_one) 
print(ent_two)
# %%
from spacy import displacy
displacy.render(doc1, style="ent", jupyter=True)
# %%
