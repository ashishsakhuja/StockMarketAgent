# 📈 Stock Market Analyst AI

This AI agent analyzes historical stock data, scrapes recent financial news, and uses AI reasoning to provide a forecast and actionable investment recommendations — powered by [CrewAI](https://github.com/joaomdmoura/crewai).

---

## Agent Capabilities

- Analyzes stock trends and moving averages
- Scrapes the latest financial news headlines
- Predicts future ROI and stock price using linear regression
- Recommends whether to Buy, Hold, or Sell (with confidence %)

---

## 🛠️ Built With

- Python 3.10+
- [CrewAI](https://github.com/joaomdmoura/crewai)
- OpenAI GPT-4
- yfinance
- LangChain
- scikit-learn

---

## 🚀 How It Works

1. **Input:** Stock ticker and time horizon (e.g., `AMD`, 2 years)
2. **Agents:**
   - 📊 `Stock Data Analyst` – fetches and analyzes historical stock data
   - 📰 `Financial News Analyst` – scrapes recent news headlines
   - 🧠 `Investment Advisor` – synthesizes insights and provides final recommendations
3. **Output:** Summary, ROI forecast, price prediction, and confidence-backed investment advice
