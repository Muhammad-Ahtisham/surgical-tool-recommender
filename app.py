import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from typing import List, Tuple

# --------------------------
# Data Loading and Caching
# --------------------------
@st.cache_data
def load_data(user_file: str, tools_file: str) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
    """
    Load and preprocess user purchase history and tool metadata.
    
    Args:
        user_file (str): Path to the Excel file containing user purchases.
        tools_file (str): Path to the Excel file containing product metadata.

    Returns:
        Tuple of rules DataFrame, product code list, and product metadata DataFrame.
    """
    df_users = pd.read_excel(user_file)
    df_tools = pd.read_excel(tools_file)

    transactions = df_users['previousPurchases'].dropna().apply(lambda x: x.split('|')).tolist()

    encoder = TransactionEncoder()
    transaction_array = encoder.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(transaction_array, columns=encoder.columns_)

    frequent_itemsets = apriori(df_encoded, min_support=0.2, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

    return rules, list(encoder.columns_), df_tools


# --------------------------
# Recommendation Logic
# --------------------------
def recommend_products(purchased_items: List[str], rules_df: pd.DataFrame, top_n: int = 5) -> List[str]:
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


def get_product_details(recommendations: List[str], df_tools: pd.DataFrame) -> pd.DataFrame:
    """
    Match recommended product codes to tool metadata.

    Args:
        recommendations (List[str]): List of recommended product codes.
        df_tools (pd.DataFrame): Product metadata DataFrame.

    Returns:
        pd.DataFrame: Filtered product info.
    """
    filtered_tools = df_tools[df_tools['Title'].str.contains('|'.join(recommendations), case=False)]
    return filtered_tools


# --------------------------
# Streamlit Web Interface
# --------------------------
st.set_page_config(page_title="Surgical Tool Recommender", layout="centered")
st.title("📈 Surgical Tool Recommendation System")

USER_FILE = "surgical_tool_recommendation_users.xlsx"
TOOLS_FILE = "Tools_2.xlsx"

rules, product_list, df_tools = load_data(USER_FILE, TOOLS_FILE)

selected_tools = st.multiselect("Select previously purchased tools:", product_list)

if st.button("Get Recommendations"):
    if selected_tools:
        recs = recommend_products(selected_tools, rules)
        if recs:
            st.success("Recommended Products:")
            detailed = get_product_details(recs, df_tools)
            if not detailed.empty:
                for _, row in detailed.iterrows():
                    st.markdown(f"### [{row['Title']}]({row['Title_URL']})")
                    st.image(row['Image'], width=150)
                    st.write(f"**Price:** {row['Price']}  ")
                    st.write("---")
            else:
                st.info("Recommendations found, but no detailed info available in Tools_2.xlsx.")
        else:
            st.info("No strong recommendations found for the selected tools.")
    else:
        st.warning("Please select at least one tool to get recommendations.")
