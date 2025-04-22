from stock_market_agent.crew import build_crew

if __name__ == "__main__":
    stock_ticker = input("📈 Enter stock ticker (Ex: AAPL, TSLA, NVDA): ").upper()
    years = input("📅 Enter number of years you're interested in (e.g. 0.5, 1, 5): ")

    print("\n🧠 Analyzing...\n")
    try:
        years = float(years)
        crew = build_crew(stock_ticker, years)
        result = crew.kickoff(inputs={"stock_ticker": stock_ticker, "years": years})
        print("\n\n**📢 Disclaimer**\nThe information provided is for educational purposes only and does not constitute "
              "financial advice. Use at your own risk.")
        print("\n💡 Final Investment Insight:\n")
        print(result)
    except ValueError:
        print("❌ Invalid input for years. Please enter a number like 1, 2, or 0.5.")
