# import packages
import streamlit as st
import pandas as pd
import os
import plotly.express as px
import openai
from dotenv import load_dotenv

# Helper  function to get dataset path
def get_dataset_path():
    # Get the current script directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to the CSV file
    csv_path = os.path.join(current_dir, "..", "..", "..", "data", "customer_reviews.csv")
    return csv_path

# Helper function to get sentiment score using OpenAI
def get_Sentiment_10(review):
    prompt = f"Assign a sentiment score from 1 to 10 for the following customer review, where 1 is very negative and 10 is very positive.:\n\n{review}\n\nSentiment Score (a number):"
    response = client.responses.create(
        model="gpt-5-mini",
        input=prompt,
        reasoning={"effort": "low"},
        text={"verbosity":"low"}
    )
    score_text = response.output_text.strip()
    try:
        score = int(score_text)
        if 1 <= score <= 10:
            return score
        else:
            return None
    except ValueError:
        return None


# Load environment variables from .env file
load_dotenv()
# Set OpenAI API key
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.title("Avelanche Customer Reviews Sentiment Analysis")
st.write("This app analyzes customer reviews using OpenAI's GPT model to determine sentiment scores.")

# Layout with two columns
col1, col2 = st.columns(2)

with col1:
    if st.button("Load Data"):
        try:
            csv_path = get_dataset_path()
            # st.session_state['df'] = pd.read_csv(csv_path)
            # TESTING PURPOSES ONLY - LOAD A SMALLER DATASET
            st.session_state['df'] = pd.read_csv(csv_path).head() # Load only first 5 rows for faster testing
            st.success("Data loaded successfully!")
        except FileNotFoundError:
            st.error("CSV file not found. Please ensure the file exists at the specified path.")

with col2:
    if st.button("Analyze Sentiments"):
        if 'df' in st.session_state:
            # simple if for prototyping
            # TODO: calculate sentiment scores where not already calculated (but not by column name)
            if 'Sentiment_10' not in st.session_state['df'].columns: # Avoid re-computation if already done
                # add sentiment scores
                with st.spinner("Analyzing sentiments..."):
                    st.session_state['df']['Sentiment_10'] = st.session_state['df']['SUMMARY'].apply(get_Sentiment_10)
                    st.success("Sentiment analysis completed successfully!")
            else:
                st.info("Sentiment scores have already been computed.")
        else:
            st.warning("Please load the data first by clicking 'Load and Process Data'.")

# Display the dataframe if loaded
if 'df' in st.session_state:
    st.subheader("All Customer Reviews Data")
    st.dataframe(st.session_state['df'])

    if 'Sentiment_10' in st.session_state['df'].columns:
        st.subheader("Sentiment Score Distribution")
        fig = px.histogram(
            st.session_state['df'], 
            x='Sentiment_10', 
            nbins=10, 
            title='Distribution of Sentiment Scores',
            labels={'Sentiment_10': 'Sentiment Score', 'count': 'Number of Reviews'})
        fig.update_layout(
            xaxis_title='Sentiment Score',
            yaxis_title='Number of Reviews',
            # assure x axis shows ALL scores from 1 to 10
            xaxis=dict(tickmode='linear', dtick=1),
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
