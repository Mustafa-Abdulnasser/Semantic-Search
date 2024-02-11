# In[3]:


import pandas as pd
from operator import itemgetter


# In[5]:


class MovieSearch:
    def __init__(self, tokenizer, dictionary, movie_tfidf_model, movie_lsi_model, movie_index, df):
        # Initialize MovieSearch with necessary components
        self.tokenizer = tokenizer  # Tokenizer object
        self.dictionary = dictionary  # Gensim Dictionary object
        self.movie_tfidf_model = movie_tfidf_model  # Gensim TF-IDF model
        self.movie_lsi_model = movie_lsi_model  # Gensim LSI model
        self.movie_index = movie_index  # Gensim MatrixSimilarity index
        self.df = df  # DataFrame containing movie information

    def extract_hot_keywords_lsi(self, article_text, top_n=5):
        # Tokenize, convert to bag-of-words, calculate TF-IDF, and transform to LSI
        article_tokens = self.tokenizer.custom_tokenizer(article_text)
        article_bow = self.dictionary.doc2bow(article_tokens)
        article_tfidf = self.movie_tfidf_model[article_bow]
        article_lsi = self.movie_lsi_model[article_tfidf]

        # Extract and sort LSI weights
        term_weights_lsi = [(self.dictionary[word_id], lsi_weight) for word_id, lsi_weight in article_lsi]
        term_weights_lsi.sort(key=itemgetter(1), reverse=True)

        # Create DataFrame for LSI weights and take top N keywords
        lsi_df = pd.DataFrame(term_weights_lsi, columns=['Term', 'LSI Weight'])
        lsi_top_keywords = lsi_df.head(top_n)
        lsi_top_keywords.fillna(0, inplace=True)

        return lsi_top_keywords

    def search_similar_movies(self, search_term):
        # Tokenize, convert to bag-of-words, calculate TF-IDF, and transform to LSI for the search term
        query_bow = self.dictionary.doc2bow(self.tokenizer.custom_tokenizer(search_term))
        query_tfidf = self.movie_tfidf_model[query_bow]
        query_lsi = self.movie_lsi_model[query_tfidf]

        # Perform semantic search and sort the results
        self.movie_index.num_best = 3
        movies_list = self.movie_index[query_lsi]
        movies_list.sort(key=itemgetter(1), reverse=True)

        # Prepare movie information list for the search results
        movie_info_list = []

        for j, movie in enumerate(movies_list):
            movie_title = self.df['movie_title'][movie[0]]
            movie_synopsis = self.df['synopsis'][movie[0]]
            movie_genre = self.df['genre'][movie[0]]  # Adjust based on actual column name

            # Extract hot keywords from the movie synopsis
            hot_keywords_df = self.extract_hot_keywords_lsi(movie_synopsis, top_n=10)

            # Build movie information dictionary
            movie_info = {
                'Title': movie_title,
                'Genre': movie_genre,
                'Year': self.df['year'][movie[0]],  # Adjust based on actual column name
                'Hot Keywords': hot_keywords_df['Term'].tolist(),
            }

            movie_info_list.append(movie_info)

            if j == (self.movie_index.num_best - 1):
                break

        return movie_info_list