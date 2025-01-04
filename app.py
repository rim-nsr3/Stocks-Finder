import streamlit as st
from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import yfinance as yf
import json
import requests

# Load environment variables from .env file
load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "stocks"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="gcp", region="us-east-1")
    )
pinecone_index = pc.Index(index_name)

# Initialize Embedding Model
model = SentenceTransformer('all-mpnet-base-v2')  # Model outputs embeddings of dimension 384

# Load Stock Metadata
with open('company_tickers.json') as f:
    stock_metadata = json.load(f)

# Generate Response Function
def generate_response(query):
    try:
        # Encode the query using the embedding model
        query_embedding = model.encode(query)

        # Query Pinecone for similar stock descriptions
        top_matches = pinecone_index.query(
            vector=query_embedding.tolist(),
            top_k=10,
            include_metadata=True,
            namespace="stock-descriptions"
        )

        # Extract relevant contexts
        contexts = [match['metadata'].get('text', '') for match in top_matches['matches']]

        # Augment the query with relevant contexts
        augmented_query = (
            "<CONTEXT>\n"
            + "\n\n-------\n\n".join(contexts[:10])
            + "\n-------\n</CONTEXT>\n\nMY QUESTION:\n" + query
        )

        # Query Groq API for a response
        groq_api_key = os.getenv("GROQ_API_KEY")
        groq_url = "https://api.groq.com/openai/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {groq_api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "llama-3.1-70b-versatile",
            "messages": [
                {"role": "system", "content": "You are a financial assistant helping to find relevant stocks based on user queries."},
                {"role": "user", "content": augmented_query},
            ]
        }

        response = requests.post(groq_url, headers=headers, json=payload)

        # Parse the response
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content'].strip()
        else:
            return f"Error: {response.status_code} - {response.text}"

    except Exception as e:
        return f"An error occurred: {e}"

# Fetch Stock Data by Ticker
import requests

def fetch_real_time_data_alpha_vantage(ticker):
    api_key = os.getenv("alpha_vantage_api_key")  # Replace with your API key
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={ticker}&apikey={api_key}"
    response = requests.get(url)
    data = response.json()

    if "Global Quote" in data:
        quote = data["Global Quote"]
        return {
            "Symbol": quote.get("01. symbol", "Unknown"),
            "Open": quote.get("02. open", "N/A"),
            "High": quote.get("03. high", "N/A"),
            "Low": quote.get("04. low", "N/A"),
            "Price": quote.get("05. price", "N/A"),
            "Volume": quote.get("06. volume", "N/A"),
            "Previous Close": quote.get("08. previous close", "N/A"),
        }
    else:
        return {"Error": "No data found for ticker using Alpha Vantage."}

def get_stock_data(ticker):
    try:
        # Ensure ticker is uppercase
        ticker = ticker.upper()

        # Fetch metadata from JSON
        json_data = None
        for key, value in stock_metadata.items():
            if value.get("ticker") == ticker:
                json_data = value
                break

        # Fetch real-time data from Alpha Vantage
        real_time_data = fetch_real_time_data_alpha_vantage(ticker)

        # Combine JSON metadata with real-time data
        stock_info = {
            "Company Name": json_data.get("title", "Unknown") if json_data else "Unknown",
            "CIK": json_data.get("cik_str", "Unknown") if json_data else "Unknown",
        }

        if "Error" not in real_time_data:
            stock_info.update(real_time_data)
        else:
            stock_info["Message"] = real_time_data["Error"]

        return stock_info

    except Exception as e:
        return f"Error fetching stock data: {e}"


# Streamlit UI
st.title("Stock Chatbot")
st.write("Search for stocks based on natural language queries or retrieve real-time data for specific tickers.")

# Query Input
query = st.text_input("Enter your query about stocks (e.g., 'What companies build data centers?'):")

# Ticker Input
ticker = st.text_input("Enter a stock ticker (e.g., AAPL, TSLA, META) for real-time data:")

# Submit Button for Query
if st.button("Submit"):
    if query:
        response = generate_response(query)
        st.write("Chatbot's Response:")
        st.success(response)
    else:
        st.warning("Please enter a query.")

# Get Stock Data Button
if st.button("Get Stock Data"):
    if ticker:
        stock_data = get_stock_data(ticker)
        if isinstance(stock_data, str):  # If the result is an error or message
            st.warning(stock_data)
        else:
            st.write(f"Data for {ticker.upper()}:")
            for key, value in stock_data.items():
                st.write(f"{key}: {value}")
    else:
        st.warning("Please enter a stock ticker.")


