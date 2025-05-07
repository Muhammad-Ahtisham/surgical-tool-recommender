
import streamlit as st
import pandas as pd
from typing import List, Tuple
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# --------------------------
# Data Loading and Caching
# --------------------------
@st.cache_data
def load_data(user_file: str, tools_file: str) -> Tuple[pd.DataFrame, List[str], pd.DataFrame, pd.DataFrame]:
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
def recommend_by_apriori(purchased_items: List[str], rules_df: pd.DataFrame, top_n: int = 5) -> List[str]:
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


def recommend_by_similarity(target_purchases: List[str], df_users: pd.DataFrame, top_n: int = 5) -> List[str]:
    target_set = set(target_purchases)
    similarities = []

    for _, row in df_users.iterrows():
        other_purchases_str = row['previousPurchases']
        if pd.isna(other_purchases_str):
            continue
        other_set = set(other_purchases_str.split('|'))
        if not other_set or other_set == target_set:
            continue

        intersection = target_set.intersection(other_set)
        union = target_set.union(other_set)
        similarity = len(intersection) / len(union)

        if similarity > 0:
            similarities.append((similarity, other_set))

    similarities.sort(reverse=True, key=lambda x: x[0])
    recommended_items = []

    for sim, other_set in similarities:
        for item in other_set:
            if item not in target_set and item not in recommended_items:
                recommended_items.append(item)
            if len(recommended_items) >= top_n:
                break
        if len(recommended_items) >= top_n:
            break

    return recommended_items


def hybrid_recommendation(purchased_items: List[str], rules_df: pd.DataFrame, df_users: pd.DataFrame, top_n: int = 5) -> List[str]:
    recs_apriori = recommend_by_apriori(purchased_items, rules_df, top_n * 2)
    recs_similar = recommend_by_similarity(purchased_items, df_users, top_n * 2)

    hybrid = []
    seen = set(purchased_items)

    for item in recs_apriori:
        if item not in seen:
            hybrid.append(item)
            seen.add(item)
        if len(hybrid) >= top_n:
            break

    for item in recs_similar:
        if item not in seen:
            hybrid.append(item)
            seen.add(item)
        if len(hybrid) >= top_n:
            break

    return hybrid


def get_product_details(recommendations: List[str], df_tools: pd.DataFrame) -> pd.DataFrame:
    filtered_tools = df_tools[df_tools['Title'].apply(lambda x: any(code.lower() in x.lower() for code in recommendations))]
    return filtered_tools


# --------------------------
# Streamlit Web Interface
# --------------------------
st.set_page_config(page_title="Surgical Tool Recommender", layout="centered")
st.title("🔬 Surgical Tool Recommendation System (Hybrid AI)")

USER_FILE = "surgical_tool_recommendation_users.xlsx"
TOOLS_FILE = "Tools_2.xlsx"

rules, product_list, df_users, df_tools = load_data(USER_FILE, TOOLS_FILE)


user_ids = df_users['userID'].dropna().unique().tolist()
specific_user_id = st.selectbox("Select User ID:", sorted(user_ids))
user_data = df_users[df_users['userID'] == specific_user_id]

purchased = []
if not user_data.empty:
    purchase_str = user_data.iloc[0]['previousPurchases']
    if pd.notna(purchase_str):
        purchased = purchase_str.split('|')

if st.button("🔍 Get Hybrid Recommendations"):
    if purchased:
        recs = hybrid_recommendation(purchased, rules, df_users)
        if recs:
            st.success(f"Hybrid Recommendations for {specific_user_id}:")
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
            st.warning("No recommendations found using hybrid method.")
    else:
        st.warning("No purchase history found for the specified user.")
