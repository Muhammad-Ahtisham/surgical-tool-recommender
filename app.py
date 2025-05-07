import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from typing import List, Tuple

# --------------------------
# Data Loading and Caching
# --------------------------
@st.cache_data
def load_data(user_file: str, tools_file: str) -> Tuple[pd.DataFrame, List[str], pd.DataFrame, pd.DataFrame]:
    """
    Load and preprocess user purchase history and tool metadata.

    Args:
        user_file (str): Path to the Excel file containing user purchases.
        tools_file (str): Path to the Excel file containing product metadata.

    Returns:
        Tuple of rules DataFrame, product code list, user data, and product metadata DataFrame.
    """
    df_users = pd.read_excel(user_file)
    df_tools = pd.read_excel(tools_file)

    transactions = df_users['previousPurchases'].dropna().apply(lambda x: x.split('|')).tolist()

    encoder = TransactionEncoder()
    transaction_array = encoder.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(transaction_array, columns=encoder.columns_)

    frequent_itemsets = apriori(df_encoded, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.5)

    return rules, list(encoder.columns_), df_users, df_tools


# --------------------------
# Recommendation Logic
# --------------------------
def recommend_products(purchased_items: List[str], rules_df: pd.DataFrame, top_n: int = 5) -> List[str]:
    purchased_set = set(purchased_items)
    matches = []

    for _, row in rules_df.iterrows():
        antecedents = set(row['antecedents']) if isinstance(row['antecedents'], frozenset) else row['antecedents']
        consequents = set(row['consequents']) if isinstance(row['consequents'], frozenset) else row['consequents']
        if antecedents.issubset(purchased_set):
            matches.append((consequents, row['confidence'], row['lift']))

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


def get_product_details(recommendations: List[str], df_tools: pd.DataFrame) -> pd.DataFrame:
    """
    Match recommended product codes to tool metadata.

    Args:
        recommendations (List[str]): List of recommended product codes.
        df_tools (pd.DataFrame): Product metadata DataFrame.

    Returns:
        pd.DataFrame: Filtered product info.
    """
    filtered_tools = df_tools[df_tools['Title'].apply(lambda x: any(code.lower() in x.lower() for code in recommendations))]
    return filtered_tools


# --------------------------
# Streamlit Web Interface
# --------------------------
st.set_page_config(page_title="Surgical Tool Recommender", layout="centered")
st.title("📈 Surgical Tool Recommendation System")

USER_FILE = "surgical_tool_recommendation_users.xlsx"
TOOLS_FILE = "Tools_2.xlsx"

rules, product_list, df_users, df_tools = load_data(USER_FILE, TOOLS_FILE)

specific_user_id = "user_123"  # <-- Set your specific user ID here
user_data = df_users[df_users['user_id'] == specific_user_id]
purchased = []
if not user_data.empty:
    purchase_str = user_data.iloc[0]['previousPurchases']
    if pd.notna(purchase_str):
        purchased = purchase_str.split('|')

if st.button("Get Recommendations for Specific User"):
    if purchased:
        recs = recommend_products(purchased, rules)
        if recs:
            st.success(f"Recommended Products for {specific_user_id}:")
            detailed = get_product_details(recs, df_tools)
            if not detailed.empty:
                for _, row in detailed.iterrows():
                    st.markdown(f"### [{row['Title']}]({row['Title_URL']})")
                    st.image(row['Image'], width=150)
                    st.write(f"**Price:** {row['Price']}  ")
                    st.write("---")
            else:
                st.warning("No metadata found for recommended items. Displaying raw codes:")
                for item in recs:
                    st.write(f"- {item}")
        else:
            st.warning("No association rules matched this user's purchase history.")
    else:
        st.warning("No purchase history found for the specified user.")
