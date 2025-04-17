import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from langchain_community.chat_models import ChatOpenAI

from stock_market_agent.tools.scrape import scrape_tool
from stock_market_agent.tools.utils import get_stock_data, get_stock_news, estimate_roi, forecast_price_linear

load_dotenv()

def build_crew(stock_ticker: str, years: float):
    llm = ChatOpenAI(model_name="gpt-4", openai_api_key=os.getenv("OPENAI_API_KEY"))

    # Agents
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

    # Tasks
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

    return Crew(
        agents=[stock_data_analyst, news_analyst, investment_advisor],
        tasks=[fetch_stock_data, fetch_news, forecast_task, provide_recommendation],
        verbose=False
    )
