import streamlit as st
from agno.agent import Agent
from agno.models.google import Gemini
from agno.embedder.google import GeminiEmbedder
from agno.tools.yfinance import YFinanceTools
import yfinance as yf
import os

stock = yf.Ticker("TATAMOTORS.NS")
print(stock.history(period="1d"))

st.title("AI Investment Agent 📈🤖")
st.caption("This app allows you to compare the performance of two stocks and generate detailed reports.")

# Fetch API key from environment variable
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Validate API key availability
if not gemini_api_key:
    st.error("Gemini API key is not set. Please configure it in the backend.")
else:
    assistant = Agent(
        model=Gemini(id="gemini-1.5-flash", api_key=gemini_api_key),
        tools=[
            YFinanceTools(
                stock_price=True,
                analyst_recommendations=True,
                stock_fundamentals=True
            )
        ],
        show_tool_calls=True,
        description="You are an investment analyst that researches stock prices, analyst recommendations, and stock fundamentals.",
        instructions=[
            "Format your response using markdown and use tables to display data where possible."
        ],
    )

    col1, col2 = st.columns(2)
    with col1:
        stock1 = st.text_input("Enter first stock symbol (e.g. AAPL)")
    with col2:
        stock2 = st.text_input("Enter second stock symbol (e.g. MSFT)")

    if stock1 and stock2:
        with st.spinner(f"Analyzing {stock1} and {stock2}..."):
            query = (
                f"Compare both the stocks - {stock1} and {stock2} "
                "and make a detailed report for an investment trying "
                "to invest and compare these stocks"
            )
            response = assistant.run(query, stream=False)
            st.markdown(response.content)
