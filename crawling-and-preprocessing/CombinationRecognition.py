from ast import literal_eval
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder
# Load file

# Create a dataframe (combination_df) to add the place from which the attractions originate.
# We used ast.literal_eval() to evaluate the literal and convert it into a list
main_file = pd.read_csv('content/data_prep_2805_2.csv')
main_file['nouns_cleaned'] = main_file['nouns_cleaned'].apply(literal_eval)
all_transactions = main_file['nouns_cleaned']
full_texts = main_file['full_text']
combination_df = pd.DataFrame(columns=['support', 'itemsets', 'length', 'place'])

# Full text is broken down into single sentences and saved in a list
# It is checked whether the attractions occur in the sentences

all_sentences_finding = []
for text, trasactions, row in zip(full_texts, all_transactions, main_file.itertuples()):
    place = row.place
    sentences = text.split('.')
    contained_word = []
    for sentence in sentences:
        founded_sent_words = []
        for transaction in trasactions:
            if transaction in sentence:
                founded_sent_words.append(transaction)
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
combination_df.to_csv('content/combinations.csv', index=False)
