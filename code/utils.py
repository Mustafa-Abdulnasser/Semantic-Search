# In[ ]:


import spacy
import re
import string
from spacy.lang.en.stop_words import STOP_WORDS
import pandas as pd

class Utils:
    def __init__(self):
        # Initialize Utils with spaCy NLP, punctuations, and stop words
        self.spacy_nlp = spacy.load("en_core_web_sm")
        self.punctuations = string.punctuation
        self.stop_words = STOP_WORDS

    def custom_tokenizer(self, sentence):
        # Custom tokenizer for processing sentences
        # Remove distracting single quotes
        sentence = re.sub('\'', '', sentence)

        # Remove digits and words containing digits
        sentence = re.sub('\w*\d\w*', '', sentence)

        # Replace extra spaces with a single space
        sentence = re.sub(' +', ' ', sentence)

        # Remove unwanted lines starting from special characters
        sentence = re.sub(r'\n: \'\'.*', '', sentence)
        sentence = re.sub(r'\n!.*', '', sentence)
        sentence = re.sub(r'^:\'\'.*', '', sentence)

        # Remove non-breaking new line characters
        sentence = re.sub(r'\n', ' ', sentence)

        # Remove punctuations
        sentence = re.sub(r'[^\w\s]', ' ', sentence)

        # Creating token object
        tokens = self.spacy_nlp(sentence)

        # Lower, strip, and lemmatize
        tokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in tokens]

        # Remove stopwords, punctuations, and exclude words less than 2 characters
        tokens = [word for word in tokens if word not in self.stop_words and word not in self.punctuations and len(word) > 2]

        # Return tokens
        return tokens

    def drop_duplicates_by_column(self, df, column_to_check):
        # Check for duplicates before dropping
        duplicates_before = df[df.duplicated(subset=column_to_check, keep='first')]
        print("Duplicates before:")
        display(duplicates_before)

        # Drop duplicates based on the specified column
        df_no_duplicates = df.drop_duplicates(subset=column_to_check, keep='first')

        # Reset index
        df_no_duplicates.reset_index(drop=True, inplace=True)

        # Display the result
        print("\nDataFrame without Duplicates:")
        display(df_no_duplicates)

        # Check for duplicates after dropping
        duplicates_after = df_no_duplicates[df_no_duplicates.duplicated(subset=column_to_check, keep=False)]
        print("\nDuplicates after:")
        display(duplicates_after)

        # Check the counts
        total_rows_original = len(df)
        total_rows_filtered = len(df_no_duplicates)
        total_duplicates_before = len(duplicates_before)
        total_duplicates_after = len(duplicates_after)

        print(f"\nTotal Rows in Original DataFrame: {total_rows_original}")
        print(f"Total Rows in Filtered DataFrame: {total_rows_filtered}")
        print(f"Total Duplicates Before: {total_duplicates_before}")
        print(f"Total Duplicates After: {total_duplicates_after}")

        return df_no_duplicates