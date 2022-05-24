#%%
#imports
from xml.dom.expatbuilder import DOCUMENT_NODE
import nltk
import pandas as pd
import spacy 
import en_core_web_sm
nlp = en_core_web_sm.load()
from spacy import displacy
from pathlib import Path

#%%
#function to display basic entity info:
def show_ents(doc): 
    if doc.ents: 
        for ent in doc.ents: 
            print(ent.text+' - ' +str(ent.start_char) +' - '+ str(ent.end_char) +' - '+ent.label_+ ' - '+str(spacy.explain(ent.label_))) 
    else: 
        print('No named entities found.')


#%%
#load location token csvs
#spacy nlp to get NEs
#concat new data into imported df
#export to new csv

import_directory = 'content/processed_places'
export_directory = 'content/ner_places/'
for file in Path(import_directory).glob('*.csv'):
    df_location = pd.read_csv(file)
    doc = " ".join(df_location['token'])
    doc = nlp(doc)

    #build lists to append
    ent_text = []
    ent_iob = []
    ent_type = []
    ent_desc = []

    i = 0
    for token in doc:
        ent_text.append(doc[i].text)
        ent_iob.append(doc[i].ent_iob_)
        ent_type.append(doc[i].ent_type_)
        ent_desc.append(spacy.explain(doc[i].ent_type_))
        i = i + 1

        #test print
        #output = [doc[i].text, doc[i].ent_type_, doc[i].ent_iob_, spacy.explain(doc[i].ent_type_)]
        #print(output)

    df_NER = df=pd.DataFrame({'ent_text':ent_text, 'ent_iob':ent_iob, 'ent_type':ent_type, 'ent_desc':ent_desc})
    df_export = pd.concat([df_location, df_NER], axis=1)#ignore_index=True)
    df_export

    #export filename
    fileName = 'ner_' + file.name
    df_export.to_csv(export_directory + fileName, index=False)


# %%
#various testlines
#location csv imports

#read location token csv
df_location_token = pd.read_csv('content/processed_places/5_The_Great_Plains.csv')
df_location_token

#read crawled text csv
df_location_text = pd.read_csv('content/crawled_rough_guides.csv')
df_location_text

#%%
#create lists of token 
#read pos-tags into list
pos_tags = df_location_token['pos_tags']
pos_tags

#read row 5 description into string
description_text = df_location_text.iloc[4,6]
description_text

#test with token from preprocessed data
doc1 = " ".join(df_location_token['token'])
doc1 = nlp(doc1)
show_ents(doc1)
#test print of IOB-Labels
ent_one = [doc1[340].text, doc1[340].ent_iob_, doc1[340].ent_type_] 
print(ent_one) 

#test with raw description string from crawled places
doc_description = nlp(description_text)
show_ents(doc_description)
#test print of IOB-Labels
ent_test = [doc_description[10].text, doc_description[10].ent_iob_, doc_description[10].ent_type_]
print(ent_test)


#%%
#visualize NER spacy
displacy.render(doc1, style="ent", jupyter=True)
displacy.render(doc_description, style="ent", jupyter=True)