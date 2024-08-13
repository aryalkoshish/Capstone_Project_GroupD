import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error,classification_report
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import GridSearchCV
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import warnings

warnings.filterwarnings('ignore')

product_df = pd.read_csv('/Users/shweta/Downloads/archive (7)/product_info.csv')
review_df_02 = pd.read_csv('/Users/shweta/Downloads/archive (7)/reviews_250-500.csv', index_col=0,
                           dtype={'author_id': 'str'})
review_df_03 = pd.read_csv('/Users/shweta/Downloads/archive (7)/reviews_500-750.csv', index_col=0,
                           dtype={'author_id': 'str'})
review_df_04 = pd.read_csv('/Users/shweta/Downloads/archive (7)/reviews_750-1250.csv', index_col=0,
                           dtype={'author_id': 'str'})
review_df_05 = pd.read_csv('/Users/shweta/Downloads/archive (7)/reviews_1250-end.csv', index_col=0,
                           dtype={'author_id': 'str'})

# MERGIG ALL REVIEWS DATAFRAMES
review_df = pd.concat([review_df_02, review_df_03, review_df_04, review_df_05], axis=0)

# CHECKING COLUMNS THAT ARE COMMON IN BOTH DATAFRAMES
cols_to_use = product_df.columns.difference(review_df.columns)
cols_to_use = list(cols_to_use)
cols_to_use.append('product_id')
print(cols_to_use)

# AS DATAFRAMES HAVE COMMON COLUMN 'product_id', WE CAN MERGE THEM ON 'product_id'
df = pd.merge(review_df, product_df[cols_to_use], how='outer', on=['product_id', 'product_id'])
df = df.iloc[:200000]
cols = """variation_desc
sale_price_usd
value_price_usd
child_max_price
child_min_price
review_title"""
cols_list = cols.split("\n")
df.drop(columns=cols_list, axis=1, inplace=True)

# DROP ROWS WITH MISSING VALUES
df.dropna(axis=0, inplace=True)

df.drop(columns=['submission_time'], axis=1, inplace=True)

# ONE-HOT ENCODING CATEGORICAL VARIABLES
categorical_columns = ['skin_tone', 'eye_color', 'hair_color', 'primary_category', 'secondary_category', 'size',
                       'tertiary_category', 'variation_type', 'variation_value', 'skin_type']
df = pd.get_dummies(df, columns=categorical_columns)

df_aggregated = df.groupby(['author_id', 'product_id']).agg({'rating': 'mean'}).reset_index()

user_item_matrix = df_aggregated.pivot(index='author_id', columns='product_id', values='rating').fillna(0)

interaction_matrix = csr_matrix(user_item_matrix.values)
n_components = min(50, interaction_matrix.shape[1])  # Ensure n_components <= n_features
svd = TruncatedSVD(n_components=n_components, random_state=42)
user_matrix = svd.fit_transform(interaction_matrix)
product_matrix = svd.components_.T

print("User IDs:")
print(user_item_matrix.index.tolist())

print("\nProduct IDs:")
print(user_item_matrix.columns.tolist())


def recommend_products(user_id, user_matrix, product_matrix, user_item_matrix, top_n=10):
    user_index = user_item_matrix.index.get_loc(user_id)
    user_ratings = user_matrix[user_index]
    scores = user_ratings.dot(product_matrix.T)
    product_indices = scores.argsort()[::-1][:top_n]
    recommended_product_ids = user_item_matrix.columns[product_indices]
    return df[df['product_id'].isin(recommended_product_ids)][['product_id', 'product_name']].drop_duplicates()


tfidf = TfidfVectorizer(stop_words='english')
product_descriptions = df.groupby('product_id')['review_text'].apply(lambda x: " ".join(x)).reset_index()
tfidf_matrix = tfidf.fit_transform(product_descriptions['review_text'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

product_indices = pd.Series(product_descriptions.index, index=product_descriptions['product_id'])


def recommend_similar_products(product_id, product_indices, cosine_sim=cosine_sim, top_n=10):
    idx = product_indices[product_id]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n + 1]
    product_indices = [i[0] for i in sim_scores]
    recommended_products = product_descriptions.iloc[product_indices]['product_id']
    return df[df['product_id'].isin(recommended_products)][['product_id', 'product_name']].drop_duplicates()


def hybrid_recommendation(user_id, product_id, user_matrix, product_matrix, user_item_matrix, product_indices,
                          cosine_sim, top_n=10):
    # Collaborative Filtering Recommendations
    if user_id in user_item_matrix.index:
        collab_recommendations = recommend_products(user_id, user_matrix, product_matrix, user_item_matrix, top_n)
    else:
        collab_recommendations = pd.DataFrame(columns=['product_id', 'product_name'])
    # Content-Based Recommendations
    if product_id in product_indices.index:
        content_recommendations = recommend_similar_products(product_id, product_indices, cosine_sim, top_n)
    else:
        content_recommendations = pd.DataFrame(columns=['product_id', 'product_name'])

    # Combine results
    combined_recommendations = pd.concat([collab_recommendations, content_recommendations]).drop_duplicates().head(
        top_n)
    return combined_recommendations


# Example usage
user_id = '965993294'
product_id = 'P411365'
recommendations = recommend_products(user_id, user_matrix, product_matrix, user_item_matrix)
print(f"Recommendations for User ID {user_id}:")
print(recommendations)
print(f"Recommendations for similar products to Product ID {product_id}:")
print(recommend_similar_products(product_id, product_indices))

test_data = pd.read_csv('/Users/shweta/Downloads/archive (7)/reviews_0-250.csv', index_col=0,
                           dtype={'author_id': 'str'})


def evaluate_model(user_matrix, product_matrix, user_item_matrix, test_data):
    test_data = test_data.drop_duplicates(subset=['author_id', 'product_id'])
    # Predict ratings for the test data
    test_user_item_matrix = test_data.pivot(index='author_id', columns='product_id', values='rating').fillna(0)
    test_user_item_matrix = test_user_item_matrix.reindex(index=user_item_matrix.index,
                                                          columns=user_item_matrix.columns, fill_value=0)

    interaction_matrix = csr_matrix(test_user_item_matrix.values)
    predicted_ratings = user_matrix.dot(product_matrix.T)

    # Calculate MSE and RMSE
    mse = mean_squared_error(interaction_matrix.toarray(), predicted_ratings)
    rmse = np.sqrt(mse)

    print(f'Mean Squared Error: {mse}')
    print(f'Root Mean Squared Error: {rmse}')

    return mse, rmse

mse, rmse = evaluate_model(user_matrix, product_matrix, user_item_matrix, test_data)

# Load pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Create product descriptions from the dataset
product_descriptions = df.groupby('product_id').agg({
    'review_text': ' '.join,
    'product_name': 'first'
}).reset_index()

# Encode product descriptions using BERT
encoded_input = tokenizer(product_descriptions['review_text'].tolist(), padding=True, truncation=True, return_tensors='pt')

# Get BERT embeddings for product descriptions
with torch.no_grad():
    embeddings = model(**encoded_input).last_hidden_state[:, 0, :].numpy()

# Example user input for a recommendation
user_input = "I am looking for a hydrating moisturizer that is long-lasting."
user_encoded_input = tokenizer(user_input, padding=True, truncation=True, return_tensors='pt')

with torch.no_grad():
    user_embedding = model(**user_encoded_input).last_hidden_state[:, 0, :].numpy()

# Calculate cosine similarity between user input and product descriptions
similarity_scores = cosine_similarity(user_embedding, embeddings)

# Get the most similar product
most_similar_index = similarity_scores.argmax()
recommended_product = product_descriptions.iloc[most_similar_index]['product_id']

print("Recommended Product ID:")
print(recommended_product)

# Get the product name from the product_info dataframe
recommended_product_name = df.loc[df['product_id'] == recommended_product, 'product_name'].values[0]
print(f"Recommended Product Name: {recommended_product_name}")

# Initialize the VADER sentiment intensity analyzer
sia = SentimentIntensityAnalyzer()

# Assuming 'review_text' is the column in your DataFrame containing the text of the reviews
df['sentiment_score'] = df['review_text'].apply(lambda x: sia.polarity_scores(x)['compound'])

def label_sentiment(score):
    if score >= 0.05:
        return 'positive'
    elif score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

# Create a new column 'sentiment' with the labels
df['sentiment'] = df['sentiment_score'].apply(label_sentiment)

def label_sentiment(score):
    if score > 2:
        return 'positive'
    elif score <= 2:
        return 'negative'
    else:
        return 'neutral'

df['true_sentiment'] = df['rating'].apply(label_sentiment)

y_true = df['true_sentiment']
y_pred = df['sentiment']

# Calculate the classification report
report = classification_report(y_true, y_pred, target_names=['positive', 'neutral', 'negative'])
print("Classification Report:\n", report)

print(product_descriptions.columns)
# Define different values of k
k_values = [5, 10, 20, 50]

for k in k_values:
    # Get top-k recommendations
    top_k_indices = similarity_scores.argsort()[0][-k:]
    recommended_products = product_descriptions.iloc[top_k_indices]['product_name']

    print(f"Top {k} recommended products:")
    print(recommended_products.tolist())

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('svc', SVC())
])

# Define parameter grid
param_grid = {
    'tfidf__max_features': [5000, 10000, 20000],
    'svc__C': [0.1, 1, 10],
    'svc__kernel': ['linear', 'rbf']
}

# Initialize GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')

# Fit GridSearchCV
grid_search.fit(df['review_text'], df['true_sentiment'])

# Print best parameters
print("Best Parameters:\n", grid_search.best_params_)



