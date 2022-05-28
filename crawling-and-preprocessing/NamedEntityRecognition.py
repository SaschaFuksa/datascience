#%%
#imports
from xml.dom.expatbuilder import DOCUMENT_NODE
import nltk
import pandas as pd
import spacy
import en_core_web_trf
import xx_sent_ud_sm
import de_dep_news_trf
from spacy import displacy
from pathlib import Path
from collections import Counter

#%%
df_prep_jenny = pd.read_csv('content/results_df_prep.csv')
df_prep_jenny


#%%
#create full text column (introduction + description)
df_locations = pd.read_csv('content/crawled_rough_guides.csv')
df_locations['full_text'] = df_locations.introduction.str.cat(df_locations.description, sep=' ')
df_locations.drop('important_places', axis=1, inplace=True)
df_locations

#arrays to turn into dataframe
all_NE = []
all_NE_count = []
all_filtered_NE = []
all_NE_no_tag = []
all_non_NE = []
all_non_NE_nouns = []
all_non_NE_count = []

ne_tags = ['EVENT', 'ORG', 'GPE', 'LOC', 'PERSON', 'PRODUCT', 'WORK_OF_ART', 'LANGUAGE', 'FAC']
pos_tags = ['NOUN', 'PROPN']

nlp = en_core_web_trf.load()

for row in df_locations.itertuples():
    print('Start loop for location:' + row.place)
    doc = nlp(row.full_text)
    #temp arrays
    found_NE = []
    count_NE = []
    filtered_NE = []
    NE_no_tag = []
    non_NE = []
    non_NE_nouns = []
    non_NE_count = []

    if doc.ents:
        for ent in doc.ents:
            found_NE.append((ent.text,ent.label_))
            if ent.label_ in ne_tags:  
                #count frequency of found NE
                counted_NE = row.full_text.count(ent.text)
                count_NE.append((ent.text, counted_NE))
                #append found NE to filtered array
                filtered_NE.append((ent.text, ent.label_))
                #append only text of found ne
                NE_no_tag.append(ent.text)
        for Token in doc:
            #non NE token 
            if Token.text not in NE_no_tag:
                print('found non-NE!')
                non_NE.append(Token.text)
                if Token.pos_ in pos_tags:
                    print('found non-NE noun!')
                    non_NE_nouns.append(Token.text)
                    counted_noun = row.full_text.count(Token.text)
                    non_NE_count.append((Token.text, counted_noun))
    else:
        print('No named entities found.')
    
    #NE
    all_NE.append(found_NE)
    all_NE_count.append(count_NE)
    all_filtered_NE.append(filtered_NE)
    all_NE_no_tag.append(NE_no_tag)

    #non NE
    all_non_NE.append(non_NE)
    all_non_NE_nouns.append(non_NE_nouns)
    all_non_NE_count.append(non_NE_count)

    print('Finished location: ' + row.place)

df_NER = pd.DataFrame({
    'found_NE':all_NE, 
    'freq_NE':all_NE_count, 
    'filtered_NE':all_filtered_NE, 
    'NE_no_tag':all_NE_no_tag,
    'non_NE':all_non_NE,
    'non_NE_nouns':all_non_NE_nouns,
    'freq_nouns':all_non_NE_count
    })

df_locations = pd.concat([df_locations, df_NER], axis=1)
df_locations.to_csv('content/all_places_ner_new.csv')
df_locations

#%%
df_locations = pd.read_csv('content/all_places_ne.csv', index_col=0)
df_locations['found_NE']=df_locations['found_NE'].apply(ast.literal_eval)
for row in df_locations.itertuples():
    df_NES = pd.DataFrame(row.found_NE, columns=['text', 'tag'])
    doc = nlp(row.full_text)
    words = [token.text for token in doc if token.is_stop != True and token.is_punct != True]
    word_freq = Counter(words)
    common_words = word_freq.most_common(5)
    print (common_words)
    break

#%%
import ast
#create list of non NE
df_import = pd.read_csv('content/all_places_ne.csv', index_col=0)
df_test = pd.DataFrame()
#df_import['found_NE'] = df_import['found_NE'].str.strip('[]').str.split(',')
df_import['found_NE']=df_import['found_NE'].apply(ast.literal_eval)
for row in df_import.itertuples():
    df_NES = pd.DataFrame(row.found_NE, columns=['text', 'tag'])
    df_NES
    break
df_NES

#%%
#function to display basic entity info:
def show_ents(doc):
    if doc.ents:
        for ent in doc.ents:
            print(ent.text+' - ' +str(ent.start_char) +' - '+ str(ent.end_char) +' - '+ent.label_+ ' - '+str(spacy.explain(ent.label_)))
    else:
        print('No named entities found.')

def recognize_entities(import_directory, nlp):
    for file in Path(import_directory).glob('*.csv'):
        df_location = pd.read_csv(file)
        doc = " ".join(df_location['token'])
        doc = nlp(doc)

        #build lists to append
        ent_text = []       #token/wort
        ent_iob = []        #iob-label -> O = kein NE ; B = Anfang einer NE ; I = NE-Wort das zum vorherigen B gehört -> Zusammengesetzte NE's B+I
        ent_type = []       #NE Kategorie-Label
        ent_desc = []       #Beschreibung des NE Kategorie-Label

        i = 0
        for token in doc:
            ent_text.append(doc[i].text)
            ent_iob.append(doc[i].ent_iob_)
            ent_type.append(doc[i].ent_type_)
            ent_desc.append(spacy.explain(doc[i].ent_type_))
            i = i + 1

        df_NER =pd.DataFrame({'ent_text':ent_text, 'ent_iob':ent_iob, 'ent_type':ent_type, 'ent_desc':ent_desc})
        df_exp = pd.concat([df_location, df_NER], axis=1)

        fileName = 'ner_' + file.name
        df_exp.to_csv(export_dir + fileName, index=False)
        print('Exported csv: ',fileName)

def recognize_entities_test():
    text = 'Though cut off by the Wall for thirty years, the eastern part of the city – the Mitte district – has always been the capital’s real centre. This is the city’s main sightseeing and shopping hub and home to many of the best places to visit in Berlin. Head here for visual inspiration on things to do in Berlin .Most visitors begin their exploration on the city’s premier boulevard Unter den Linden , starting at the most famous landmark, the Brandenburg Gate , then moving over to the adjacent seat of Germany’s parliament, the Reichstag . Unter den Linden’s most important intersection is with Friedrichstrasse, which cuts north–south.At its eastern end Unter den Linden is lined by stately Neoclassical buildings and terminates on the shores of Museum Island , home to eastern Berlin’s leading museums, but its natural extension on the other side of the island is Karl-Liebknecht-Strasse, which leads to a distinctively GDR-era part of the city around Alexanderplatz , the eastern city’s main commercial and transport hub.Northwest from here, the Spandauer Vorstadt was once the heart of the city’s Jewish community, and has some fascinating reminders of those days, though today it’s best known for the restaurants, bars, boutiques and nightlife around the Hackescher Markt.Back at the Brandenburg Gate, a walk south along the edge of the gigantic Tiergarten park takes you to the swish modern Potsdamer Platz , a bustling entertainment quarter that stands on what was for decades a barren field straddling the death-strip of the Berlin Wall.Huddled beside Potsdamer Platz is the Kulturforum , an agglomeration of cultural institutions that includes several high-profile art museums. Also fringing the park are Berlin’s diplomatic and government quarters, where you’ll find some of the city’s most innovative architecture, including the formidable Hauptbahnhof.The western end of the Tiergarten park is given over to a zoo, which is also the name of the main transport hub at this end of town. This is the gateway to City West , West Berlin’s old centre and is best known for its shopping boulevards, particularly the upmarket Kurfürstendamm.Schöneberg and Kreuzberg , the two residential districts immediately south of the centre, are home to much of Berlin’s most vibrant nightlife. The former is smart and is popular as a gay area, while Kreuzberg is generally grungy and edgy.Beyond Kreuzberg’s eastern fringes, and back in what used to be East Berlin is Friedrichshain which offers some unusual architectural leftovers from the Eastern Bloc of the 1950s, while to the north Prenzlauer Berg is one of the few places in which the atmosphere of prewar Berlin has been preserved – complete with cobbled streets and ornate facades.Berlin’s eastern suburbs are typified by a sprawl of prewar tenements punctuated by high-rise developments and heavy industry, though the lakes, woodland and small towns and villages dotted around Köpenick offer a genuine break from the city.The leafy western suburbs are even more renowned for their woodland (the Grunewald) and lakes (the Havel), with more besides: attractions include the baroque Schloss Charlottenburg , with its adjacent art museums; the impressive 1930s Olympic Stadium; the Dahlem museum complex, which displays everything from German folk art to Polynesian huts; and the medieval town of Spandau.Further out, foremost among possible places to visit on day-trips are Potsdam , location of Frederick the Great’s Sanssouci palace, and the former concentration camp of Sachsenhausen , north of Berlin in Oranienburg.'
    doc = nlp(text)
    #show_ents(doc)
    relevant_buildings = []
    is_building = 'Buildings, airports, highways, bridges, etc.'
    relevant_countries = []
    is_countries = 'Countries, cities, states'
    relevant_instituions = []
    is_institution = 'Companies, agencies, institutions, etc.'
    relevant_fictionals = []
    is_fictional = 'People, including fictional'

    # Location und country rausfilter
    def show_ents_2(doc):
        if doc.ents:
            for ent in doc.ents:
                if ent.text != 'Berlin':
                    if str(spacy.explain(ent.label_)) == is_building:
                        relevant_buildings.append(ent.text)
                    elif str(spacy.explain(ent.label_)) == is_countries:
                        relevant_countries.append(ent.text)
                    elif str(spacy.explain(ent.label_)) == is_institution:
                        relevant_instituions.append(ent.text)
                    elif str(spacy.explain(ent.label_)) == is_fictional:
                        relevant_fictionals.append(ent.text)

        else:
            print('No named entities found.')

    show_ents_2(doc)
    print('relevant_buildings:')
    print(relevant_buildings)
    print('relevant_countries:')
    print(relevant_countries)
    print('relevant_instituions:')
    print(relevant_instituions)
    print('relevant_fictionals:')
    print(relevant_fictionals)


#%%
#load location token csvs
#spacy nlp to get NEs
#concat new data into imported df
#export to new csv

nlp_module = en_core_web_trf.load()
#nlp_module = xx_sent_ud_sm.load()
#nlp_module = de_dep_news_trf.load()

import_dir = 'content/processed_places'
export_dir = 'content/new_ner_places/'

df_places_ner = recognize_entities(import_directory=import_dir, nlp=nlp_module)

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

#%%
#create full text column (introduction + description)
from nltk.tokenize import word_tokenize

df_locations = pd.read_csv('content/crawled_rough_guides.csv')
df_locations['full_text'] = df_locations.introduction.str.cat(df_locations.description, sep=' ')
df_locations.drop('important_places', axis=1, inplace=True)
df_locations

all_NE = []
nlp = en_core_web_trf.load()

for row in df_locations.itertuples():
    all_stopwords = nlp.Defaults.stop_words

    text = row.full_text
    text_tokens = word_tokenize(text)
    tokens_without_sw= [word for word in text_tokens if not word in all_stopwords]
    print(tokens_without_sw)
    
    #doc = nlp(tokens_without_sw)
    #found_NE = []
    #for token in doc:
    #    print(token)
    
    break


#%%

df_example = pd.read_csv('content/processed_places/11_Mexico_City.csv')
word_freq = Counter(df_example['normed_token'])
common_words = word_freq.most_common(5)
print (common_words)
#%%
freq = []
for file in Path('content/processed_places').glob('*.csv'):
    df_place = pd.read_csv(file)
    word_freq = Counter(df_place['normed_token'])
    name = file.name.split('_')
    freq.append((name[0], word_freq))

df_freq = pd.DataFrame(freq, columns=['number', 'token_freq'])
df_freq

df_locations = pd.read_csv('content/all_places_ne.csv', index_col=0)
df_locations

df_locations_new = pd.merge(df_locations, df_freq, on=df_locations.number)
df_locations_new.to_csv('content/all_places_ne.csv')
