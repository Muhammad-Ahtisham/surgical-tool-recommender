
import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

@st.cache_data
def load_data():
    user_df = pd.read_excel("surgical_tool_recommendation_users.xlsx")
    transactions = user_df['previousPurchases'].dropna().apply(lambda x: x.split('|')).tolist()
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_trans = pd.DataFrame(te_ary, columns=te.columns_)
    frequent_itemsets = apriori(df_trans, min_support=0.2, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    return rules, list(te.columns_)

def recommend_products(purchased_items, rules_df, top_n=5):
    purchased_items = set(purchased_items)
    matches = []
    for _, row in rules_df.iterrows():
        if row['antecedents'].issubset(purchased_items):
            matches.append((row['consequents'], row['confidence'], row['lift']))
    sorted_matches = sorted(matches, key=lambda x: (x[1], x[2]), reverse=True)
    recommended = []
    for match in sorted_matches:
        for item in match[0]:
            if item not in purchased_items and item not in recommended:
                recommended.append(item)
            if len(recommended) >= top_n:
                break
        if len(recommended) >= top_n:
            break
    return recommended

st.title("Surgical Tool Recommendation System")
rules, all_products = load_data()
selected_items = st.multiselect("Select previously purchased tools:", all_products)
if st.button("Recommend"):
    if selected_items:
        recommendations = recommend_products(selected_items, rules)
        if recommendations:
            st.success("Recommended products:")
            for item in recommendations:
                st.write(f"- {item}")
        else:
            st.info("No strong recommendations found for the selected tools.")
    else:
        st.warning("Please select at least one tool.")
