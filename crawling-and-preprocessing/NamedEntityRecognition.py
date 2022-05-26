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