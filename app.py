import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder

# Load the preprocessed dataset
data = pd.read_csv(r"C:\Users\91838\Desktop\Online Retail.csv")  # Replace with your dataset file path

# Preprocess the data (encode user and product IDs)
user_encoder = LabelEncoder()
product_encoder = LabelEncoder()
data['CustomerID'] = user_encoder.fit_transform(data['CustomerID'])
data['StockCode'] = product_encoder.fit_transform(data['StockCode'])

# Build a user-item interaction matrix
utility_matrix = pd.pivot_table(data, values='Quantity', index='CustomerID', columns='StockCode', fill_value=0)

# Calculate item-item similarity (cosine similarity)
item_similarity = cosine_similarity(utility_matrix.T)

# Define a function to get recommendations
def get_recommendations(product_id, num_recommendations=5):
    similar_scores = item_similarity[product_id]
    similar_product_indices = similar_scores.argsort()[-num_recommendations-1:-1][::-1]
    recommended_products = [product_encoder.inverse_transform(idx) for idx in similar_product_indices]
    return recommended_products

# Streamlit UI
st.title("E-commerce Recommendation System")

# User input for product ID
product_id_to_recommend = st.text_input("Enter a Product ID:", "")

if product_id_to_recommend:
    try:
        product_id_to_recommend = int(product_id_to_recommend)
        recommendations = get_recommendations(product_id_to_recommend)
        st.subheader("Recommended Products:")
        for recommendation in recommendations:
            st.write(recommendation)
    except ValueError:
        st.error("Please enter a valid Product ID.")
