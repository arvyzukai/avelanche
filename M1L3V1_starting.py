# import packages
import streamlit as st
import pandas as pd
import re
import os

# Helper function to clean text
def clean_text(text):
    # Remove special characters and extra spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Helper function to get dataset path
def get_dataset_path():
    # Get the current script directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to the CSV file
    # csv_path = os.path.join(current_dir, "..", "..", "data", "customer_reviews.csv")
    csv_path = "customer_reviews.csv"
    return csv_path


st.title("Hello, GenAI!")
st.write("This is your GenAI-powered data processing app.")

# Layout with two columns
col1, col2 = st.columns(2)

with col1:
    if st.button("Load and Process Data"):
        try:
            csv_path = get_dataset_path()
            st.session_state['df'] = pd.read_csv(csv_path)
            st.success("Data loaded successfully!")
        except FileNotFoundError:
            st.error("CSV file not found. Please ensure the file exists at the specified path.")

with col2:
    if st.button("Parse data"):
        if 'df' in st.session_state:
            st.session_state['df']['cleaned_review'] = st.session_state['df']['SUMMARY'].apply(clean_text)
            st.success("Data parsed and cleaned successfully!")
        else:
            st.warning("Please load the data first by clicking 'Load and Process Data'.")

        

# Display the dataframe if loaded
if 'df' in st.session_state:
    st.subheader("Filter by product")
    product = st.selectbox("Select a product:", ["All Products"] + list(st.session_state['df']['PRODUCT'].unique()))
    st.subheader("Customer Reviews Data")
    if product != "All Products":
        filtered_df = st.session_state['df'][st.session_state['df']['PRODUCT'] == product]
    else:
        filtered_df = st.session_state['df']

    st.dataframe(filtered_df)

    st.subheader("Sentiment Score by Product")
    # grouped_df = filtered_df.groupby('PRODUCT')['SENTIMENT_SCORE'].mean()
    grouped_df = st.session_state['df'].groupby('PRODUCT')['SENTIMENT_SCORE'].mean()
    st.bar_chart(grouped_df)

    import plotly.express as px
    st.subheader("Sentiment Score Distribution")
    fig = px.histogram(
        filtered_df, 
        x='SENTIMENT_SCORE', 
        nbins=10, 
        title='Sentiment Score Distribution',
        labels={'SENTIMENT_SCORE': 'Sentiment Score', 
                'count': 'Number of Reviews'})
    fig.update_layout(
        xaxis_title='Sentiment Score',
        yaxis_title='Number of Reviews',
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

    import altair as alt
    st.subheader("Sentiment Score by Product (Altair)")
    alt_chart = alt.Chart(filtered_df).mark_bar().encode(
        x=alt.X('PRODUCT:N', title='Product'),
        y=alt.Y('mean(SENTIMENT_SCORE):Q', title='Average Sentiment Score'),
        color=alt.Color('PRODUCT:N', legend=None)
    ).properties(
        title='Average Sentiment Score by Product'
    )
    st.altair_chart(alt_chart, use_container_width=True)

    
