# 📈 Sales Index Forecasting For Business Growth
Abstract:
Developed a machine learning-based sales index forecasting system using FISERV data, achieving 90% accuracy. Deployed a Streamlit web app for real-time predictions across 25+ businesses, driving data-informed strategies and improved decision-making.


Overview:
This project delivers a powerful, data-driven system that forecasts the sales index of small businesses across U.S. metropolitan areas. Leveraging historical data from the Fiserv Small Business Index, we built and deployed machine learning models—culminating in a real-time web application for business decision-makers.

🔍 Problem Statement:
How can businesses anticipate shifts in consumer behavior and regional economic activity to make better strategic decisions?

✅ Key Objectives:
-Analyze sales and transaction trends across sectors and states.

-Predict future sales index using advanced ML models.

-Offer an interactive app to empower real-time forecasting.



📊 Data Source
Dataset: Fiserv Small Business Index

Features Used:

-Sales & Transaction Index (SA/NSA)

-Month-over-Month (MoM) & Year-over-Year (YoY) % changes

-Sector, Sub-sector, and Region-level granularity


🧹 Data Cleaning & Engineering (PySpark)
-Missing value imputation (mean-based and row-wise dropping)

-Combined SA & NSA data for simplification

-Date parsing, column renaming, whitespace trimming

-Outlier detection (IQR method)

-Normalization using Min-Max Scaler

-One-hot encoding for categorical variables


🤖 Models Implemented
Implemented and compared six machine learning models for forecasting the sales index. In which XGboost Regressor outperformed every 
models with 90% accuracy.

🌐 Web Application (Streamlit)
A user-friendly Streamlit app allows business users to:

-Input sector, state, and key metrics

-Get real-time predictions of normalized sales index

-Make informed investment or marketing decisions


🧠 Powered by the trained XGBoost model
📂 Launch the app
streamlit run app.py


🧠 Insights
XGBoost model explains 90% of sales index variance

Sectors like Retail and Construction show high growth

States like Montana (MT) and Mississippi (MS) are top performers for specific sectors
