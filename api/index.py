import yfinance as yf
import warnings
from crewai import Agent, Task, Crew
from crewai_tools import ScrapeWebsiteTool
import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from sklearn.linear_model import LinearRegression
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd

app = FastAPI()
load_dotenv()

@app.get("/")
def root():
    return {"message": "StockMarketAgent is live"}

warnings.filterwarnings("ignore")
scrape_tool = ScrapeWebsiteTool()

class StockRequest(BaseModel):
    stock_ticker: str
    years: float

def get_stock_data(ticker, years):
    try:
        stock = yf.Ticker(ticker)
        period = f"{years}y" if years >= 1 else "6mo"
        hist = stock.history(period=period)

        if hist.empty:
            raise ValueError("No data returned from Yahoo Finance.")

        hist = hist.tail(200)
        latest_price = hist["Close"].iloc[-1]

        if years < 1:
            sma_short = hist["Close"].rolling(window=5).mean().iloc[-1]
            sma_long = hist["Close"].rolling(window=20).mean().iloc[-1]
        elif years < 3:
            sma_short = hist["Close"].rolling(window=20).mean().iloc[-1]
            sma_long = hist["Close"].rolling(window=50).mean().iloc[-1]
        else:
            sma_short = hist["Close"].rolling(window=50).mean().iloc[-1]
            sma_long = hist["Close"].rolling(window=200).mean().iloc[-1]

        short_dev = (latest_price - sma_short) / sma_short * 100
        long_dev = (latest_price - sma_long) / sma_long * 100

        summary = (
            f"Latest closing price: ${latest_price:.2f}\n"
            f"Short-term SMA: ${sma_short:.2f} ({short_dev:.2f}% deviation)\n"
            f"Long-term SMA: ${sma_long:.2f} ({long_dev:.2f}% deviation)"
        )
        return summary
    except Exception as e:
        return f"Could not retrieve stock data for {ticker}. Error: {str(e)}"

def get_stock_news(ticker):
    url = f"https://finance.yahoo.com/quote/{ticker}?p={ticker}&.tsrc=fin-srch"
    response = scrape_tool.run(website_url=url)
    response = response[:1000]
    headlines = [line.strip() for line in response.split("\n") if line.strip()][:3]
    summary = "Latest headlines: " + ", ".join(headlines) if headlines else "No recent news found."
    return {"output": summary, "source": url}

def estimate_roi(ticker, years_ahead=5):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="5y")
    if hist.empty:
        return "No data available."

    hist = hist.tail(200)
    start_price = hist['Close'].iloc[0]
    end_price = hist['Close'].iloc[-1]
    annual_return = ((end_price / start_price) ** (1/5)) - 1
    future_value = end_price * ((1 + annual_return) ** years_ahead)

    return f"Estimated ROI: {annual_return*100:.2f}% per year\n" \
           f"Projected price in {years_ahead} years: ${future_value:.2f}"

def forecast_price_linear(ticker, years_ahead=5):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="10y")
    if hist.empty:
        return "No data available."

    hist = hist.reset_index().tail(300)
    hist['Days'] = (hist['Date'] - hist['Date'].min()).dt.days
    X = hist[['Days']]
    y = hist['Close']

    model = LinearRegression().fit(X, y)
    future_day = hist['Days'].max() + (years_ahead * 365)
    predicted_price = model.predict([[future_day]])[0]

    predictions = model.predict(X)
    residuals = y - predictions
    std_error = np.std(residuals)
    confidence_margin = 1.96 * std_error

    lower_bound = predicted_price - confidence_margin
    upper_bound = predicted_price + confidence_margin

    return (
        f"Linear Regression Forecast for {ticker} in {years_ahead} years: ${predicted_price:.2f}\n"
        f"95% Confidence Interval: ${lower_bound:.2f} - ${upper_bound:.2f}"
    )

@app.post("/analyze")
def analyze_stock(request: StockRequest):
    openai_api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(model_name="gpt-4", openai_api_key=openai_api_key)

    stock_data_analyst = Agent(
        role="Stock Data Analyst",
        goal="Retrieve and analyze stock market data, identifying trends and trading signals.",
        backstory="Experienced stock analyst specializing in technical and fundamental analysis.",
        llm=llm,
        verbose=False,
        allow_delegation=False
    )

    news_analyst = Agent(
        role="Financial News & Earnings Call Analyst",
        goal="Summarize earnings call reports and news articles to extract key insights.",
        backstory="Expert in financial journalism, summarizing corporate reports for investors.",
        llm=llm,
        tools=[scrape_tool],
        verbose=False,
        allow_delegation=False
    )

    investment_advisor = Agent(
        role="Investment Advisor",
        goal="Analyze all data sources and provide stock recommendations (Buy, Hold, Sell).",
        backstory="Portfolio strategist trained in risk management and investment forecasting.",
        llm=llm,
        verbose=False,
        allow_delegation=False
    )

    fetch_stock_data = Task(
        description="Retrieve stock data for {stock_ticker}, analyze trends, and detect key signals based on a {years}-year horizon.",
        expected_output="Stock price trends, moving averages, and performance deviations based on investment horizon.",
        agent=stock_data_analyst,
        function=lambda inputs: get_stock_data(inputs["stock_ticker"], inputs["years"])
    )

    fetch_news = Task(
        description="Scrape financial news and summarize key insights for {stock_ticker}.",
        expected_output="Summary of latest news and earnings call reports.",
        agent=news_analyst,
        function=lambda inputs: get_stock_news(inputs["stock_ticker"])
    )

    forecast_task = Task(
        description="Forecast future price and ROI of {stock_ticker} for the next {years} years.",
        expected_output="Predicted price and annualized ROI using historical data.",
        agent=investment_advisor,
        function=lambda inputs: estimate_roi(inputs["stock_ticker"], int(inputs["years"])) + "\n" +
                                 forecast_price_linear(inputs["stock_ticker"], int(inputs["years"]))
    )

    provide_recommendation = Task(
        description="Based on the stock data and news analysis for a {years}-year outlook, determine if {stock_ticker} is a Buy, Hold, or Sell.",
        expected_output="Final investment decision with reasoning. Explicitly mention Buy, Hold, or Sell. Explicitly mention confidence level percentage. Explicitly mention a price prediction. Explicitly mention any sources used at the end, this includes website links.",
        agent=investment_advisor
    )

    stock_market_crew = Crew(
        agents=[stock_data_analyst, news_analyst, investment_advisor],
        tasks=[fetch_stock_data, fetch_news, forecast_task, provide_recommendation],
        verbose=False
    )

    result = stock_market_crew.kickoff(inputs={"stock_ticker": request.stock_ticker, "years": request.years})
    return {
        "ticker": request.stock_ticker,
        "result": result
    }
