# In[1]:


# Import necessary libraries
import warnings
warnings.filterwarnings("ignore")

import numpy as np # Linear algebra
import pandas as pd # Data processing, CSV file I/O (e.g., pd.read_csv)
import spacy
import string
import gensim
import operator
import re
from spacy.lang.en.stop_words import STOP_WORDS
from gensim import corpora, models
from operator import itemgetter
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
import torch
from yake import KeywordExtractor


# In[2]:


class Word2VecSearch:
    def __init__(self, word2vec_model_path, tokenizer):
        # Initialize Word2VecSearch with the Word2Vec model and tokenizer
        self.word2vec_model = Word2Vec.load(word2vec_model_path)
        self.tokenizer = tokenizer
        self.spacy_nlp = spacy.load("en_core_web_sm")

    def get_word2vec_embeddings(self, text):
        # Get Word2Vec embeddings using the custom tokenizer
        tokens = self.tokenizer.custom_tokenizer(text)
        embeddings = [self.word2vec_model.wv[word] for word in tokens if word in self.word2vec_model.wv.key_to_index]
        return np.mean(embeddings, axis=0) if embeddings else np.zeros(self.word2vec_model.vector_size)
    
    def extract_hot_keywords_word2vec(self, synopsis, top_keywords=5):
        # Tokenize and extract Word2Vec embeddings and similarities
        tokens = [token.text.lower() for token in self.spacy_nlp(synopsis)]
        unique_tokens = list(set(tokens))
        embeddings = [self.word2vec_model.wv[word] for word in unique_tokens if word in self.word2vec_model.wv]
        mean_embedding = np.mean(embeddings, axis=0) if embeddings else np.zeros(self.word2vec_model.vector_size)
        similarities = np.dot(embeddings, mean_embedding) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(mean_embedding))
        ranked_words_indices = np.argsort(similarities)[::-1][:top_keywords]
        hot_keywords = [(unique_tokens[i], similarities[i]) for i in ranked_words_indices]
        return hot_keywords    

    def extract_hot_keywords_word2vec_yake(self, synopsis, top_keywords, max_ngram_size=3, deduplication_threshold=0.4):
        # Tokenize and extract keywords using YAKE
        tokens = self.tokenizer.custom_tokenizer(synopsis)
        text_for_yake = " ".join(tokens)
        extractor = KeywordExtractor(lan="en", top=top_keywords, n=max_ngram_size, dedupLim=deduplication_threshold)
        keywords = extractor.extract_keywords(text_for_yake)
        hot_keywords = [keyword for keyword, _ in keywords]
        return hot_keywords

    def semantic_search_word2vec(self, query_title, df, yake, top_movies=3, top_keywords=5):
        # Check if the query title is in the DataFrame
        if query_title not in df['movie_title'].values:
            print(f"Movie with title '{query_title}' not found. Searching based on movie title with other movies' synopses.")
            query_document = query_title
        else:
            query_document = df[df['movie_title'] == query_title]['synopsis'].values[0]

        # Get embeddings for the query document
        query_embeddings = self.get_word2vec_embeddings(query_document)

        # Get embeddings for trained documents
        trained_documents = df['synopsis'].tolist()
        trained_document_embeddings = np.array([self.get_word2vec_embeddings(doc) for doc in trained_documents])

        # Calculate similarities and rank movies
        similarities = np.dot(trained_document_embeddings, query_embeddings) / (
            np.linalg.norm(trained_document_embeddings, axis=1) * np.linalg.norm(query_embeddings)
        )

        # Rank movies and extract the most similar ones
        ranked_movies_indices = np.argsort(similarities)[::-1][1:top_movies + 1]
        most_similar_movies = df.iloc[ranked_movies_indices]

        if yake:
            # Extract hot keywords for each movie using YAKE
            hot_keywords_list = []
            for index, row in most_similar_movies.iterrows():
                movie_synopsis = row['synopsis']
                hot_keywords = self.extract_hot_keywords_word2vec_yake(movie_synopsis, top_keywords=top_keywords)
                hot_keywords_list.append(hot_keywords)

            most_similar_movies['Hot Keywords'] = hot_keywords_list

        return most_similar_movies