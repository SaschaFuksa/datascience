from ast import literal_eval

import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder

# Load file
# Create a dataframe (combination_df) to add the place from which the attractions originate.
# We used ast.literal_eval() to evaluate the literal and convert it into a list

main_file = pd.read_csv('content/data_prep_2805_3.csv')
main_file['singular_cleaned_nound'] = main_file['singular_cleaned_nound'].apply(literal_eval)
all_attractions = main_file['singular_cleaned_nound']
full_texts = main_file['full_text']
combination_df = pd.DataFrame(columns=['support', 'itemsets', 'length', 'place'])

# Full text is broken down into single sentences and saved in a list
# It is checked whether the attractions occur in the sentences

all_sentences_finding = []
for text, attractions, row in zip(full_texts, all_attractions, main_file.itertuples()):
    place = row.place
    sentences = text.split('.')
    contained_word = []
    for sentence in sentences:
        founded_sent_words = []
        for attraction in attractions:
            if attraction.lower() in sentence:
                founded_sent_words.append(attraction)
        if founded_sent_words:
            contained_word.append(founded_sent_words)
    all_sentences_finding.append(contained_word)

    # Apply One-hot-encoding
    te = TransactionEncoder()
    te_ary = te.fit(contained_word).transform(contained_word)
    te_df = pd.DataFrame(te_ary, columns=te.columns_)

    # Create a PANDAS Dataframe named "frequent_itemsets" using the apriori agorithm, minimum support=0.0045 and show
    # real columnames
    frequent_itemsets = apriori(te_df, min_support=0.0045, use_colnames=True, max_len=2)
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
    frequent_df = frequent_itemsets[(frequent_itemsets['length'] == 2) &
                                    (frequent_itemsets['support'] >= 0.01)]
    combination_df = combination_df.append(frequent_df)
    combination_df = combination_df.fillna(place)

combination_df_without_frozensets = pd.DataFrame(columns=['support', 'itemsets', 'place'])
for row in combination_df.itertuples():
    first_attraction, second_attraction = row.itemsets
    combination = first_attraction + ' ' + second_attraction
    combination_df_without_frozensets = combination_df_without_frozensets.append(
        {'support': row.support, 'itemsets': combination, 'place': row.place},
        ignore_index=True)

combination_df_without_frozensets.to_csv('content/combinations_2.csv', index=False)
