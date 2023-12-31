import streamlit as st
import pandas as pd


df = pd.read_csv("GroceryStoreDataSet.csv")

itemsets = df['MILK,BREAD,BISCUIT'].apply(lambda x: set(x.split(',')))
transactions = list(itemsets)
unique_items = list(set(item for sublist in itemsets for item in sublist))
transaction_matrix = pd.DataFrame(0, columns=unique_items, index=range(len(transactions)))


  
st.title('Analisis Transaksi ')
selected_item = st.selectbox('Pilih produk:', unique_items)
selected_item_counts = transaction_matrix[transaction_matrix[selected_item] == 1].sum()
total_transactions = len(transaction_matrix)
percentage_likelihood = (selected_item_counts / total_transactions) * 100

  
percentage_likelihood = percentage_likelihood[percentage_likelihood.index != selected_item]

st.markdown("<h2 style='text-align: left; color: white; font-size: 14px;'>Jika pelanggan membeli produk tersebut maka kemungkinan membeli produk di bawah secara bersamaan</h2>", unsafe_allow_html=True)

max_percentage_item = percentage_likelihood.idxmax()
max_percentage_value = percentage_likelihood.max()
css_style = "background-color: green; padding: 10px; border-radius: 10px; max-width: 200px; margin: left;"

st.markdown(
        f"<div style='{css_style}'>"
        f"<p style='text-align: left; color: white;margin-left:63px; font-size: 16px;'>{max_percentage_item}</p>"
        "</div><br>",
        unsafe_allow_html=True,
    )



