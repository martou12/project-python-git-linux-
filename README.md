# project-python-git-linux-# 📈 Python, Git, Linux for Finance - Dashboard

**Project:** Collaborative Financial Dashboard for Asset & Portfolio Management.  
**Course:** Python for Finance / DevOps  
**Live URL:** [http://13.39.50.124:8501](http://13.39.50.124:8501)  
**Status:** 🟢 Online (24/7 Deployment)

---

## 👥 Team Members (Division of Work)

| Module | Student Name | Responsibility |
| :--- | :--- | :--- |
| **Quant A** | [CARRIERE Simon ] | **Single Asset Analysis**: Univariate analysis, backtesting strategies (Momentum/Buy-and-Hold), and predictive models. |
| **Quant B** | [VERSCHELDE Martin] | **Portfolio Management**: Multi-asset simulation, correlation matrix, diversification metrics, and optimization. |

---

## 🚀 Features

### 1. Interactive Dashboard (Streamlit)
- **Real-time Data**: Retrieves financial data dynamically via APIs.
- **Auto-Refresh**: Data is automatically updated every 5 minutes.
- **Visualizations**: Interactive charts for price history, strategy performance, and drawdowns.

### 2. Automation (Cron Job) 
A daily financial report is generated automatically on the Linux server to track volatility and performance metrics.
- **Frequency**: Every day at **20:00 (8 PM)**.
- **Output**: JSON reports stored locally in `reports/`.
- **Metrics**: Volatility, Open/Close prices, Max Drawdown.

**Server-side Configuration (Crontab):**
```bash
00 20 * * * cd /home/ubuntu/project-python-git-linux- && /usr/bin/python3 scripts/daily_report.py >> /home/ubuntu/project-python-git-linux-/reports/cron_log.txt 2>&1