import streamlit as st
from textblob import TextBlob
import plotly.graph_objects as go
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import pandas as pd
import re


import concurrent.futures

# Streamlit app configuration must be called first
st.set_page_config(page_title="Product Sentiment Analysis", page_icon="\U0001F6D2", layout="wide")

# Ensure VADER is downloaded
nltk.download('vader_lexicon', quiet=True)

# Load the Excel file and cache the data to reduce load times
@st.cache_data
def load_excel_data(file_path):
    excel_data = pd.ExcelFile(file_path)
    sheet_data = {}
    for sheet_name in excel_data.sheet_names:
        sheet_data[sheet_name] = excel_data.parse(sheet_name)
    return sheet_data

file_path = r'D:\FINAL_PROJECT_1\dataset\all_sites_review.xlsx'
sheet_data = load_excel_data(file_path)

# Get the list of all products from all sheets
all_products = []
for df in sheet_data.values():
    all_products.extend(df['ProductName'].dropna().unique().tolist())
all_products = sorted(set(all_products))  # Remove duplicates and sort

# Function to fetch reviews based on product name from the Excel data
@st.cache_data
def fetch_reviews_from_excel(product_name, platform=None):
    platform_sheets = {
        "Amazon": "Amazon",
        "Flipkart": "Flipkart",
        "Ajio": "ajio",
        "Meesho": "misho"
    }
    
    reviews = []
    product_name = re.escape(product_name.lower().strip())  # Convert to lowercase and escape special characters

    def fetch_reviews_from_sheet(sheet_name):
        df = sheet_data[sheet_name]
        product_reviews = df[df['ProductName'].str.lower().str.contains(product_name, na=False, regex=True)]
        return product_reviews['Review'].tolist()
    
    if platform and platform in platform_sheets:
        sheet_name = platform_sheets[platform]
        reviews = fetch_reviews_from_sheet(sheet_name)
    else:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_sheet = {executor.submit(fetch_reviews_from_sheet, sheet): sheet for sheet in platform_sheets.values()}
            for future in concurrent.futures.as_completed(future_to_sheet):
                reviews.extend(future.result())
    
    return reviews

# Function to analyze sentiment using VADER and TextBlob
def analyze_sentiment_combined(reviews):
    sia = SentimentIntensityAnalyzer()
    sentiments = {"positive": 0, "neutral": 0, "negative": 0}
    polarity_scores = []
    subjectivity_scores = []

    for review in reviews:
        vader_score = sia.polarity_scores(review)
        polarity_scores.append(vader_score['compound'])

        if vader_score['compound'] >= 0.05:
            sentiments["positive"] += 1
        elif -0.05 < vader_score['compound'] < 0.05:
            sentiments["neutral"] += 1
        else:
            sentiments["negative"] += 1
        
        textblob_analysis = TextBlob(review)
        subjectivity_scores.append(textblob_analysis.sentiment.subjectivity)

    average_polarity = sum(polarity_scores) / len(polarity_scores) if polarity_scores else 0
    average_subjectivity = sum(subjectivity_scores) / len(subjectivity_scores) if subjectivity_scores else 0
    buy_probability = min(max((average_polarity + 1) / 2, 0), 1) * 100

    if sentiments['positive'] > sentiments['negative'] and sentiments['positive'] > sentiments['neutral']:
        overall_sentiment = "Positive"
    elif sentiments['negative'] > sentiments['positive'] and sentiments['negative'] > sentiments['neutral']:
        overall_sentiment = "Negative"
    else:
        overall_sentiment = "Neutral"

    return sentiments, buy_probability, average_subjectivity, overall_sentiment

# Custom CSS to style the tabs and other elements
st.markdown("""
    <style>
        /* General Styling */
        .main {
            background: linear-gradient(to right, #ffecd2,);
            padding: 20px;
        }
        h1 {
            color: #ff5733;
            font-size: 3rem;
            text-align: center;
            font-family: 'Arial Black', sans-serif;
        }
        .stButton>button {
            background-color: #007bff;
            color: white;
            border-radius: 10px;
            font-size: 18px;
            padding: 10px;
            border: none;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
        }
        .stButton>button:hover {
            background-color: #0056b3;
        }
        .footer {
            position: fixed; 
            left: 0; 
            bottom: 0; 
            width: 100%; 
            background-color: #343a40; 
            color: white; 
            text-align: center; 
            padding: 10px; 
            font-family: 'Arial', sans-serif.
        }
        /* Styling for Streamlit Tabs */
        [data-baseweb="tab"] {
            font-size: 1.5rem; /* Increase font size */
            padding: 15px 25px; /* Increase padding */
        }
        [data-baseweb="tab"] span {
            font-size: 1.5rem; /* Ensure icons scale with text */
        }
        [data-baseweb="tab"]:hover {
            background-color: #ffcccc !important; /* Highlight on hover */
        }
    </style>
""", unsafe_allow_html=True)

# Application Header
st.markdown("<h1>\U0001F6D2 Welcome to Product Sentiment Analysis \U0001F6C5</h1>", unsafe_allow_html=True)
st.markdown("**Analyze customer reviews to gauge overall sentiment and provide a buy recommendation.**")

# Tabs for different e-commerce platforms
tabs = st.tabs(["\U0001F6D2Amazon", "\U0001F6CD️Flipkart", "\U0001F45EAjio", "\U0001F4E6Meesho"])

platforms = ["Amazon", "Flipkart", "Ajio", "Meesho"]

for idx, tab in enumerate(tabs):
    with tab:
        platform = platforms[idx]
        st.markdown(f"### {platform} Product Review Analysis")

        # Dropdown selection for products
        selected_product = st.selectbox(
            f"Select a Product from {platform} (Optional)", 
            [""] + all_products,  # Add empty option for optionality
            key=f"dropdown_{platform}"
        )
        
        user_input = st.text_input(f"Or Enter Product Name on {platform}", key=f"text_input_{platform}")

        # Determine which product name to use
        product_name = user_input.strip() if user_input.strip() else selected_product

        if st.button(f"Analyze {platform} Reviews", key=f"button_{platform}"):
            if product_name:
                with st.spinner('Fetching reviews... \U0001F575️‍♂️'):
                    reviews = fetch_reviews_from_excel(product_name, platform)

                if reviews:
                    st.success(f"✅ Extracted {len(reviews)} reviews for '{product_name}' on {platform}.")
                    sentiments, buy_probability, average_subjectivity, overall_sentiment = analyze_sentiment_combined(reviews)

                    st.markdown(f"<h3>Overall Sentiment Analysis: {overall_sentiment}</h3>", unsafe_allow_html=True)
                    st.markdown(f"<p><strong>Positive Reviews:</strong> {sentiments['positive']}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p><strong>Neutral Reviews:</strong> {sentiments['neutral']}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p><strong>Negative Reviews:</strong> {sentiments['negative']}</p>", unsafe_allow_html=True)

                    # Bar chart for sentiment distribution
                    st.markdown(f"<h4>Sentiment Distribution</h4>", unsafe_allow_html=True)
                    fig_bar = go.Figure()
                    fig_bar.add_trace(go.Bar(
                        x=['Positive', 'Neutral', 'Negative'],
                        y=[sentiments['positive'], sentiments['neutral'], sentiments['negative']],
                        marker_color=['#1f77b4', '#ff7f0e', '#2ca02c']  # Custom colors for each bar
                    ))
                    fig_bar.update_layout(
                        title="Sentiment Distribution",
                        xaxis_title="Sentiment",
                        yaxis_title="Number of Reviews",
                        margin=dict(t=50, b=100),  # Adjust the bottom margin to avoid overlap
                        annotations=[
                            dict(
                                text="This bar chart shows the distribution of positive, neutral, and negative reviews.",
                                xref="paper", yref="paper",
                                x=0.5, y=-0.3, showarrow=False,  # Move the annotation lower to avoid overlap
                                font=dict(size=12)
                            )
                        ]
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

                    # Pie chart for probability distribution
                    st.markdown(f"<h4>Sentiment Probability Distribution</h4>", unsafe_allow_html=True)
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=['Positive', 'Neutral', 'Negative'], 
                        values=[sentiments['positive'], sentiments['neutral'], sentiments['negative']],
                        marker=dict(colors=['#1f77b4', '#ff7f0e', '#2ca02c'])  # Custom colors for each slice
                    )])
                    fig_pie.update_layout(
                        title="Sentiment Probability Distribution",
                        annotations=[
                            dict(
                                text="This pie chart shows the proportion of positive, neutral, and negative reviews.",
                                xref="paper", yref="paper",
                                x=0.5, y=-0.2, showarrow=False,
                                font=dict(size=12)
                            )
                        ]
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

                    # Detailed Summary of Output
                    st.markdown("### Detailed Summary of Sentiment Analysis")
                    st.markdown(
                        f"**Product:** '{product_name}'\n"
                        f"**Platform:** {platform}\n"
                        f"**Total Reviews Analyzed:** {len(reviews)}\n\n"
                        f"**Sentiment Distribution:**\n"
                        f"- Positive: {sentiments['positive']} reviews ({sentiments['positive'] / len(reviews) * 100:.2f}%)\n"
                        f"- Neutral: {sentiments['neutral']} reviews ({sentiments['neutral'] / len(reviews) * 100:.2f}%)\n"
                        f"- Negative: {sentiments['negative']} reviews ({sentiments['negative'] / len(reviews) * 100:.2f}%)\n\n"
                        f"**Overall Sentiment:** {overall_sentiment}\n\n"
                        f"**Buy Probability:** {buy_probability:.2f}%\n\n"
                        f"**Average Subjectivity:** {average_subjectivity:.2f} (on a scale of 0 to 1)\n\n"
                        "### Interpretation and Recommendations:\n"
                        "- **Positive Sentiment:** With the majority of reviews being positive, it indicates that most customers are satisfied with this product. A high buy probability suggests that this product is well-received by its users.\n"
                        "- **Neutral Sentiment:** A significant proportion of neutral reviews may indicate that while the product meets basic expectations, it may not be exceptional. This could be a factor if you are considering purchasing a product with more specific or higher expectations.\n"
                        "- **Negative Sentiment:** The presence of negative reviews should not be ignored. Analyze these reviews to understand common complaints or issues. If the proportion of negative reviews is high, it might be worth reconsidering the purchase or investigating alternatives.\n"
                        "- **Subjectivity:** The subjectivity score indicates how much personal bias or opinion is present in the reviews. A higher score suggests more opinion-based reviews, which could be based on personal experiences rather than objective facts.\n"
                    )

                else:
                    st.warning(f"No reviews found for '{product_name}' on {platform}.")
            else:
                st.error("Please enter a product name.")

# Footer
st.markdown("""
    <div class="footer">
        <p><strong></strong>  2024 Product Sentiment Analysis</p>
    </div>
""", unsafe_allow_html=True)