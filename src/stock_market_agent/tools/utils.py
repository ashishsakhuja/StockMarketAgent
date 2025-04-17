import yfinance as yf
import numpy as np
from sklearn.linear_model import LinearRegression

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
    from .scrape import scrape_tool
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
