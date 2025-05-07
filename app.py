import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from typing import List, Tuple

# --------------------------
# Data Loading and Caching
# --------------------------
@st.cache_data
def load_data(file_path: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load and preprocess purchase history data for association rule mining.
    
    Args:
        file_path (str): Path to the Excel file containing purchase data.

    Returns:
        Tuple containing association rules DataFrame and list of all products.
    """
    df = pd.read_excel(file_path)
    transactions = df['previousPurchases'].dropna().apply(lambda x: x.split('|')).tolist()

    encoder = TransactionEncoder()
    transaction_array = encoder.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(transaction_array, columns=encoder.columns_)

    # Apply Apriori algorithm
    frequent_itemsets = apriori(df_encoded, min_support=0.2, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

    return rules, list(encoder.columns_)


# --------------------------
# Recommendation Logic
# --------------------------
def recommend_products(purchased_items: List[str], rules_df: pd.DataFrame, top_n: int = 5) -> List[str]:
    """
    Generate product recommendations based on association rules.

    Args:
        purchased_items (List[str]): List of products user has already purchased.
        rules_df (pd.DataFrame): DataFrame containing association rules.
        top_n (int): Number of recommendations to return.

    Returns:
        List of recommended products.
    """
    purchased_set = set(purchased_items)
    matches = []

    for _, row in rules_df.iterrows():
        if row['antecedents'].issubset(purchased_set):
            matches.append((row['consequents'], row['confidence'], row['lift']))

    sorted_matches = sorted(matches, key=lambda x: (x[1], x[2]), reverse=True)
    recommendations = []

    for consequents, _, _ in sorted_matches:
        for item in consequents:
            if item not in purchased_set and item not in recommendations:
                recommendations.append(item)
            if len(recommendations) >= top_n:
                break
        if len(recommendations) >= top_n:
            break

    return recommendations


# --------------------------
# Streamlit Web Interface
# --------------------------
st.set_page_config(page_title="Surgical Tool Recommender", layout="centered")
st.title("📈 Surgical Tool Recommendation System")

DATA_FILE = "surgical_tool_recommendation_users.xlsx"

# Load rules and products
rules, product_list = load_data(DATA_FILE)

# User input
selected_tools = st.multiselect("Select previously purchased tools:", product_list)

# Show recommendations
if st.button("Get Recommendations"):
    if selected_tools:
        results = recommend_products(selected_tools, rules)
        if results:
            st.success("Recommended Products:")
            for item in results:
                st.write(f"- {item}")
        else:
            st.info("No strong recommendations found for the selected tools.")
    else:
        st.warning("Please select at least one tool to get recommendations.")
