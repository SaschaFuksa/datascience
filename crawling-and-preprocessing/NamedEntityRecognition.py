#%%
#imports
from xml.dom.expatbuilder import DOCUMENT_NODE
import nltk
import pandas as pd
import spacy
import en_core_web_trf
from spacy import displacy
from pathlib import Path
from collections import Counter
from nltk.corpus import wordnet as wn
from ast import literal_eval
import re

#%%
#create full text column (introduction + description)
df_locations = pd.read_csv('content/crawled_rough_guides.csv')
df_locations['full_text'] = df_locations.introduction.str.cat(df_locations.description, sep=' ')
df_locations.drop('important_places', axis=1, inplace=True)
df_locations

#2D Arrays die alle Daten speichern sollen, bis diese zur crawled-csv hinzugefügt werden
all_NE = []                 #alle NEs die identifiziert wurden
all_NE_count = []           #freq der NEs
all_filtered_NE = []        #gefilterte NEs von Katergorien die wir betrachten (ne_tags) 
all_NE_no_tag = []          #alle NEs ohne zugeörigem Label
all_non_NE = []             #alle Token, die zu keiner NE gehören
all_non_NE_nouns = []       #alle Nomen, von nicht NE-Token
all_non_NE_count = []       #freq der nicht NE-Nomen

#NE Kategorien die betrachtet werden
ne_tags = ['EVENT', 'ORG', 'GPE', 'LOC', 'PERSON', 'PRODUCT', 'WORK_OF_ART', 'LANGUAGE', 'FAC']
#POS-Tags zum filtern von nicht NE-Nomen
pos_tags = ['NOUN', 'PROPN']

#ladem des englischen spacy models


#Abarbeiten der Locations mit hilfe der Schleife
#Für jede Location werden alle Arrays mit Werten befüllt
#Aus dem Text werden ausgelesen:
#   Named entities und das dazu gehörende Label
#   Bestimmen der Frequency der Named entities (NE-Text, Anzahl)
#   Named entities, welche zu den in ne_tags definierten Kategorien gehören
#   Extra Spalte nur mit der NE ohne Label
#   Alle Token, die zu keiner NE gehören
#   Alle Nomen-Token, die zu keiner NE gehören
#   Bestimmen der Frequency der Nomen-Token (Nomen-Text, Anzahl)

nlp = en_core_web_trf.load()
for row in df_locations.itertuples():
    print('location:' + row.place)
    doc = nlp(row.full_text)
    
    #temp arrays
    found_NE = []
    count_NE = []
    filtered_NE = []
    NE_no_tag = []
    non_NE = []
    non_NE_nouns = []
    non_NE_count = []

    #prüfen ob Entities erkannt wurden
    if doc.ents:
        #Schleife zum durchgehen der identifizierten NE
        for ent in doc.ents:
            #Hinzufügen von NE zur Liste
            found_NE.append((ent.text,ent.label_))
            #Prüfen ob NE zu einer der gesuchten Kategorien gehört
            if ent.label_ in ne_tags:  
                #Frequenz der NE zählen
                counted_NE = row.full_text.count(ent.text)
                count_NE.append((ent.text, counted_NE))
                #NE zur entsprechenden Liste hinzufügen
                filtered_NE.append((ent.text, ent.label_))
                #Nur den NE-Text zur entsprechenden Liste hinuzfügen
                NE_no_tag.append(ent.text)
        #Alle Token des Location-Textes durchgehen
        for Token in doc:
            #Prüfen ob Token in NEs vorkommt
            if Token.text not in NE_no_tag:
                #Token entsprechender Liste hinzufügen
                non_NE.append(Token.text)
                #Prüfen ob Token ein Nomen ist
                if Token.pos_ in pos_tags:
                    #Token entsprechender Liste hinzufügen und Freq bestimmen
                    non_NE_nouns.append(Token.text)
                    counted_noun = row.full_text.count(Token.text)
                    non_NE_count.append((Token.text, counted_noun))
    else:
        print('No named entities found.')
    
    #Alle temp-arrays den gesamt-arrays hinzufügen
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

#Dataframe aus allen arrays erstellen
df_NER = pd.DataFrame({
    'found_NE':all_NE, 
    'freq_NE':all_NE_count, 
    'filtered_NE':all_filtered_NE, 
    'NE_no_tag':all_NE_no_tag,
    'non_NE':all_non_NE,
    'non_NE_nouns':all_non_NE_nouns,
    'freq_nouns':all_non_NE_count
    })

#Location Dataframe mit neuem Dataframe joinen und exportieren
df_locations = pd.concat([df_locations, df_NER], axis=1)
df_locations.to_csv('content/all_places_ner_new.csv')
df_locations


#%%
#function to display basic entity info:
def show_ents(doc):
    if doc.ents:
        for ent in doc.ents:
            print(ent.text+' - ' +str(ent.start_char) +' - '+ str(ent.end_char) +' - '+ent.label_+ ' - '+str(spacy.explain(ent.label_)))
    else:
        print('No named entities found.')

#%%
#Hypernym von nicht NE-Nomen suchen und hinzufügen
#Der Gedanke hierbei ist, dass anhand der Hypernyme eine Gruppierung von nicht NEs stattfinden kann
#Die Hypernymen stellen anschließend die gemeinsamen Attractions einer Kategorie da z.B. Lake und Sea sind beides "Body of Water"

#laden des datensatz
df_locations = pd.read_csv('crawling-and-preprocessing/content/all_places_ner_new.csv', index_col=0)
df_locations['non_NE_nouns'] = df_locations['non_NE_nouns'].apply(literal_eval)

non_NE_tags = []   

#für jede location werden die nomen durchgegangen und das enstprechende Hypernym gesucht
#Nomen ohne Hypernym bekommen ein '-'
#Zum schluß wird eine neue Spalte angehängt und als neuer Datensatz exportiert
for row in df_locations.itertuples():
    print('Location: ' + row.place)
    #temp array
    noun_hypernym = []
    for noun in row.non_NE_nouns:
        synset = wn.synsets(noun)
        if len(synset) > 0:
            synset = synset[0]
            hypernyms = synset.hypernyms()
            if len(hypernyms) > 0:
                hypernym = hypernyms[0].lemma_names()[0]
                noun_hypernym.append((noun, hypernym))
            else:
                noun_hypernym.append((noun, '-'))
        else:
            noun_hypernym.append((noun, '-'))

    non_NE_tags.append(noun_hypernym)

    print('End loop location: ' + row.place)  

df_tags = pd.DataFrame({'non_NE_tags':non_NE_tags})
df_loactions_new = pd.concat([df_locations, df_tags], axis=1)
df_loactions_new.to_csv('content/data_prep.csv', index=False)

#%%
#Anpassung des Datensatz zur einfacheren Anbindung an Dash
#Auslesen aus Freq-Tupeln, sodass Anzahl ohne zugehöriges NE/Nomen in eigener Spalte steht

df_locations = pd.read_csv('content/data_prep.csv')
df_locations['freq_nouns'] = df_locations['freq_nouns'].apply(literal_eval)
df_locations['freq_NE'] = df_locations['freq_NE'].apply(literal_eval)

all_freq_NE = []
all_freq_nouns = []

for row in df_locations.itertuples():
    freq_NE = []
    freq_noun = []
    for namedE in row.freq_NE:
        freq_NE.append(namedE[1])
    for noun in row.freq_nouns:
        freq_noun.append(noun[1])
    
    all_freq_NE.append(freq_NE)
    all_freq_nouns.append(freq_noun)

df_freq = pd.DataFrame({'freq_NE_int':all_freq_NE, 'freq_noun_int':all_freq_nouns})
df_export = pd.concat([df_locations, df_freq], axis=1)
df_export.to_csv('content/data_prep_2805.csv')

#%%
#Anpassung des Datensatz um ungültige Nomen/Attractions zu entfernen
#Bedingungen sind:
#   Attraction muss mehr als 3 Chars lang sein
#   Jahreszahlen sind ungültig
#   Keine alleinstehenden Sonderzeichen
#   "Bar" stellt eine Ausnahme dar
df_locations = pd.read_csv('content/data_prep_2805.csv')
df_locations['non_NE_nouns'] = df_locations['non_NE_nouns'].apply(literal_eval)
df_locations['freq_noun_int'] = df_locations['freq_noun_int'].apply(literal_eval)

#REgex zum bestimmen von ungültigen Einträgen
regex = re.compile('^[a-zA-Z]+$')
all_cleaned_noun = []

#Jede Location prüfen auf ungültige Einträge
for row in df_locations.itertuples():
    print('started loop: ' + row.place)
    new_nouns = []
    for index, token in reversed(list(enumerate(row.non_NE_nouns))):   
        if regex.search(token) == None or len(token)<4:
            if token.lower() != 'bar':
                #Freq-Eintrag entfernen, da ansonsten die Zuordnung nicht mehr passt
                row.non_NE_nouns.remove(token)
                del row.freq_noun_int[index]

    new_nouns = row.non_NE_nouns
    new_nouns = [string.lower() for string in new_nouns]
    all_cleaned_noun.append(new_nouns)

df_cleaned_nouns = pd.DataFrame({'nouns_cleaned':all_cleaned_noun})
df_export = pd.concat([df_locations, df_cleaned_nouns], axis=1)
    
df_export.to_csv('crawling-and-preprocessing/content/data_prep_2805_2.csv', index=False)
df_export
