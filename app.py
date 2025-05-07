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

    # Filter rules where antecedents are a subset of purchased items
    filtered_rules = rules_df[rules_df['antecedents'].apply(lambda x: x.issubset(purchased_set))]

    sorted_rules = filtered_rules.sort_values(by=['confidence', 'lift'], ascending=False)

    recommendations = []
    for _, row in sorted_rules.iterrows():
        for item in row['consequents']:
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
st.title("\U0001F4C8 Surgical Tool Recommendation System")

USER_FILE = "surgical_tool_recommendation_users.xlsx"
TOOLS_FILE = "Tools_2.xlsx"

rules, product_list, df_users, df_tools = load_data(USER_FILE, TOOLS_FILE)

selected_tools = st.multiselect("Select previously purchased tools:", product_list)

if st.button("Get Recommendations"):
    if selected_tools:
        with st.spinner("Generating recommendations..."):
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
                    st.warning("Recommendations generated but no matching product metadata found.")
                    st.write("Raw product codes:", recs)
            else:
                st.warning("No association rules matched your selection. Try selecting more or different tools.")
    else:
        st.warning("Please select at least one tool to get recommendations.")
