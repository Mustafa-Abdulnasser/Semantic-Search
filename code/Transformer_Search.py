# In[ ]:


import torch
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from yake import KeywordExtractor


# In[ ]:

class TransformerSearch:
    def __init__(self, sentence_transformer_model):
        # Initialize TransformerSearch with the Sentence Transformer model and spaCy NLP
        self.sentence_transformer_model = SentenceTransformer(sentence_transformer_model)
        self.spacy_nlp = spacy.load("en_core_web_sm")

    def get_transformer_embeddings(self, text):
        # Get embeddings using the Sentence Transformer model
        return self.sentence_transformer_model.encode(text, convert_to_tensor=True)
    
    def extract_hot_keywords_transformer(self, synopsis, top_keywords=5):
        # Tokenize, get unique tokens, and calculate mean embedding
        tokens = [token.text.lower() for token in self.spacy_nlp(synopsis) if not token.is_stop and not token.is_punct]
        unique_tokens = list(set(tokens))
        embeddings = [self.get_transformer_embeddings(token) for token in unique_tokens]
        mean_embedding = torch.stack(embeddings).mean(dim=0)

        # Calculate cosine similarities and rank words
        similarities = torch.nn.functional.cosine_similarity(torch.stack(embeddings), mean_embedding.unsqueeze(0))
        similarities_np = similarities.cpu().numpy().reshape(-1, 1)
        ranked_words_indices = np.argsort(similarities_np.flatten())[::-1][:top_keywords]

        # Extract hot keywords
        hot_keywords = [(unique_tokens[i], similarities_np[i, 0]) for i in ranked_words_indices]
        return hot_keywords
    
    def extract_hot_keywords_transformer_yake(self, synopsis, top_keywords, max_ngram_size=3, deduplication_threshold=0.3):
        # Tokenize and extract keywords using YAKE
        tokens = [token.text.lower() for token in self.spacy_nlp(synopsis) if not token.is_stop and not token.is_punct]
        text_for_yake = " ".join(tokens)
        extractor = KeywordExtractor(lan="en", top=top_keywords, n=max_ngram_size, dedupLim=deduplication_threshold)
        keywords = extractor.extract_keywords(text_for_yake)
        hot_keywords = [keyword for keyword, _ in keywords]
        return hot_keywords

    def semantic_search_transformer(self, query_title, df, yake, top_movies=3, top_keywords=5):
        # Check if the query title is in the DataFrame
        if query_title not in df['movie_title'].values:
            print(f"Movie with title '{query_title}' not found. Searching based on movie title with other movies' synopses.")
            query_document = query_title
        else:
            query_document = df[df['movie_title'] == query_title]['synopsis'].values[0]

        # Get embeddings for the query document
        query_embeddings = self.get_transformer_embeddings(query_document)

        # Get embeddings for trained documents
        trained_documents = df['synopsis'].tolist()
        trained_document_embeddings = [self.get_transformer_embeddings(doc) for doc in trained_documents]

        # Calculate cosine similarities and rank movies
        similarities = torch.nn.functional.cosine_similarity(torch.stack(trained_document_embeddings), query_embeddings)
        ranked_movies_indices = np.argsort(similarities.cpu().numpy().flatten())[::-1][1:top_movies + 1]
        most_similar_movies = df.iloc[ranked_movies_indices]

        if yake:
            # Extract hot keywords for each movie using YAKE
            hot_keywords_list = []
            for index, row in most_similar_movies.iterrows():
                movie_synopsis = row['synopsis']
                hot_keywords = self.extract_hot_keywords_transformer_yake(movie_synopsis, top_keywords=top_keywords)
                hot_keywords_list.append(hot_keywords)

            most_similar_movies['Hot Keywords'] = hot_keywords_list

        return most_similar_movies