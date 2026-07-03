Link Tableau: https://public.tableau.com/app/profile/irvanda.nugroho/viz/Assignment_Timeseries/Dashboard1

Link Slides: https://canva.link/u45dn71iyhn5upz

# Sales Promotion Impact Analysis: Time Series Forecasting with SARIMAX 🛒📈

![Project Status](https://img.shields.io/badge/Status-Completed-success)
![Tools](https://img.shields.io/badge/Tools-Python%20%7C%20Tableau-blue)
![Domain](https://img.shields.io/badge/Domain-Retail%20%26%20Demand%20Forecasting-green)

---

## 📌 Business Overview & Problem Statement

The retail industry faces ongoing challenges in managing constantly shifting customer demand. While promotions are commonly used to boost sales, their effectiveness isn't consistent across product categories or time periods — and promotional activity itself can introduce demand fluctuations that complicate sales forecasting and inventory planning (Harsha, 2025 — *A Comprehensive Analysis of Sales Promotional Effects on the Entire Demand Life Cycle*).

This project uses **Python** (time series analysis, statistical testing, SARIMAX modeling) and **Tableau** (interactive dashboarding) to analyze the impact of promotions on sales for **Corporación Favorita**, a supermarket chain in Ecuador, and to build a forecasting model that supports more effective, data-driven promotional strategy.

> **"How much does running a promotion actually move sales, which product categories benefit the most, and when is the optimal time to run a promotion?"**

---

## 🗂️ Repository Structure

```
sales-promotion-impact-time-series-analysis
│
├── README.md                                          # Main project documentation
├── notebook/
│   └── assignment_fixed_timeseries.ipynb              # Full Python script (cleaning, stationarity testing, SARIMAX modeling)
├── data/
│   └── train.csv                                      # Raw dataset (3,000,888 rows)
├── output/
│   └── sales_forecast_output.csv                       # Weekly sales forecast (8-week horizon)
└── report/
    └── Time_Series_Promotion_Impact_Analysis.pdf        # Full presentation deck (Tableau dashboards + insights)
```

---

## 🗃️ Dataset & Schema

This project uses **1 table** — historical daily sales data from the Favorita supermarket chain in Ecuador (3,000,888 rows, 6 columns):

| Column | Description |
|--------|-------------|
| `id` | Index number (export artifact, not used in analysis) |
| `date` | Sales transaction date per store; used as the time index for time series analysis |
| `store_nbr` | Unique identifier for each Favorita store |
| `family` | Product category/family (e.g. BEVERAGES, DAIRY, GROCERY I) |
| `sales` | Total sales for a given product category, store, and date |
| `onpromotion` | Number of products in a category under an active promotion, for a given store and date |

**Key challenge identified:** At the daily level, the relationship between promotion volume and sales is noisy — a natural consequence of the time lag between when a promotion is seen and when a purchase actually happens. Aggregating to a **weekly level** dampened this noise and revealed a strong, consistent relationship (**80% correlation**) between weekly promotion volume and weekly sales, which became the basis for the forecasting approach.

---

## 💻 Tech Stack & Analysis Pipeline

Data cleaning, statistical testing, and time series modeling were performed in **Python** (Pandas, NumPy, Matplotlib, Seaborn, SciPy, Statsmodels, pmdarima), and the final interactive dashboard was built in **Tableau**.

### 1. Data Understanding & Cleaning

**Initial Inspection:**
```python
df = pd.read_csv('train.csv')
df.info()
df.duplicated().sum()                    # Result: 0 duplicates
(df.isnull().sum() / len(df)) * 100      # Result: 0% missing values
```

**Type Conversion:**
```python
# Convert date to datetime and set as time series index
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
```

**Outlier Detection:**
```python
# Outliers found in 'sales' and 'onpromotion' — retained, as they represent
# genuine spikes in real sales/promotion volume rather than data errors
```

### 2. Statistical Testing

**Normality Testing (D'Agostino-Pearson):**
```python
from scipy.stats import normaltest

def dagostino_test(data, nama_kolom):
    sample = data.sample(n=min(10000, len(data)), random_state=42)
    stat, p = normaltest(sample)
    kesimpulan = "Berdistribusi normal" if p > 0.05 else "Tidak berdistribusi normal"

dagostino_test(df['sales'], 'Sales')
dagostino_test(df['onpromotion'], 'On Promotion')
```
**Result:** Both `sales` and `onpromotion` are not normally distributed (p < 0.05).

### 3. Correlation Analysis (Daily vs. Weekly)

```python
# Daily level — noisy relationship due to promotion-to-purchase time lag
korelasi_harian = df[['onpromotion', 'sales']].corr().iloc[0, 1]

# Weekly level — noise dampened, strong relationship emerges
weekly_eda = df.resample('W').agg({'sales': 'sum', 'onpromotion': 'sum'})
weekly_eda.corr()
```
**Result:** Weekly promotion volume correlates strongly with weekly sales (**r = 0.80**), while the daily-level relationship is noticeably noisier.

### 4. Feature Engineering

| Column | Purpose |
|--------|---------|
| `Is Promo` | Splits days into promo / non-promo conditions |
| `Is Weekend` | Identifies weekend effects |
| `Average Sales Non Promo` | Average sales on days without promotion |
| `Average Sales Promo` | Average sales on days with promotion |
| `Month Number` | Identifies month |
| `Week` | Identifies week number |
| `Year` | Identifies year |

### 5. Time Series Modeling (SARIMAX)

**Stationarity Testing (Augmented Dickey-Fuller):**
```python
from statsmodels.tsa.stattools import adfuller

# Original weekly series
adf_result = adfuller(weekly_sales)
# ADF Statistic: -1.91 | p-value: 0.33 -> NOT stationary

# After first-order differencing
adf_result_diff = adfuller(weekly_sales.diff().dropna())
# ADF Statistic: -4.77 | p-value: 0.00006 -> Stationary
```

**Parameter Identification (ACF/PACF):**
```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(weekly_sales_diff, lags=20)
plot_pacf(weekly_sales_diff, lags=20)
# P (AR order) = 1, Q (MA order) = 1
```

**Model Comparison (Train/Test Split 80:20, 241 weeks total):**
```python
import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Models compared: ARIMA Manual (1,1,1), Auto ARIMA, SARIMA (1,1,0)x(1,0,0,52), SARIMAX (+onpromotion)
model_sarimax = SARIMAX(train['sales'], exog=train['onpromotion'],
                         order=(1,1,1), seasonal_order=(1,0,0,52))
results = model_sarimax.fit()
```

| Model | RMSE | MAE | MAPE |
|-------|------|-----|------|
| **SARIMAX (+ onpromotion)** | **566,700.09** | **477,230.72** | **7.98%** |
| Auto ARIMA | 670,039.10 | 571,527.31 | 9.30% |
| SARIMA (1,1,0)x(1,0,0,52) | 710,658.10 | 616,992.21 | 10.06% |
| ARIMA Manual (1,1,1) | 738,651.49 | 606,454.39 | 9.82% |

**SARIMAX** was selected as the final model — adding `onpromotion` as an exogenous variable meaningfully improved forecast accuracy over models using only historical sales.

### 6. Tableau Dashboard

An interactive dashboard was built for reusable reporting, combining:
- **Sales Overview:** Top 10 sales & promotion volume by product family, weekly sales trend, promotion-vs-sales scatter plot.
- **Promotion Deep Dive:** Average sales by family (promo vs. non-promo), average sales by day of week, monthly sales by family, and an 8-week SARIMAX sales forecast.

---

## 🔍 Key Insights & Findings

### 1. 📈 Consistent Growth with a Repeating Seasonal Pattern

From 2013 to 2017, Corporación Favorita's weekly sales showed a **consistent upward growth trend**, with a recurring seasonal pattern each year — spikes mid-year and toward year-end, likely tied to Ecuador's holiday and celebration periods. This repeating pattern provided a strong foundation for building a time-based forecasting model like SARIMAX.

### 2. 🛒 Two Categories Dominate Total Revenue

**GROCERY I** dominates total sales by a significant margin ($343.5M), followed by **BEVERAGES** ($217.0M) — together far outpacing the other 8 categories combined. This concentration makes these two categories the top priority for promotional strategy, since their performance drives the largest share of total revenue.

| Rank | Category | Total Sales |
|------|----------|-------------|
| 1 | GROCERY I | $343,462,735 |
| 2 | BEVERAGES | $216,954,486 |
| 3 | PRODUCE | $122,704,685 |
| 4 | CLEANING | $97,521,289 |
| 5 | DAIRY | $64,487,709 |

### 3. 🎯 Promotion Genuinely Lifts Sales — Confirmed Statistically

At the weekly level, promotion volume correlates strongly with sales (**r = 0.80**). For most categories, **average daily sales are meaningfully higher on promo days than non-promo days** — e.g. GROCERY I averages $4,411 on promo days vs. $2,718 without. The SARIMAX model confirms this statistically: each additional promoted item per week contributes **+20.5 units** to sales (coefficient = 20.5, p = 0.001).

| Category | Avg. Sales (Promo) | Avg. Sales (Non-Promo) |
|----------|--------------------|------------------------|
| GROCERY I | $4,411 | $2,718 |
| BEVERAGES | $3,215 | $1,292 |
| PRODUCE | $2,438 | $792 |
| CLEANING | $1,240 | $868 |
| DAIRY | $933 | $480 |

### 4. 🗓️ December is Peak Demand — Timing Promotions Matters

GROCERY I sales climb steadily from a January low ($3,420) to a December peak ($4,896) — a **28% jump from November to December alone**, and roughly **43% higher than the lowest month**. Promotions run around or during December are likely to have the largest impact, since they align with naturally elevated demand; January–February promotions are likely less efficient given naturally low demand.

### 5. 📅 Weekends Outperform Weekdays by 33–63%

Average daily sales peak on **Sunday ($463)** and **Saturday ($433)**, while **Thursday is the lowest ($284)**. Weekdays consistently underperform weekends, making Saturday and Sunday the optimal days to activate or intensify promotions for maximum consumer exposure.

### 6. 🔮 8-Week Forecast Confirms a Measurable Promotion Uplift

The SARIMAX model projects that maintaining promotions at the historical average intensity (32,332 items/week) results in consistently higher total sales versus a no-promotion scenario across the entire 8-week forecast horizon (Aug–Oct 2017) — an estimated **+734,077 total unit sales**, or **+91,760 units/week** on average.

---

## 💡 Strategic Recommendations

**1. Time Promotions Around Naturally High Demand**
Run higher-intensity promotions in December and the year-end period, when demand is naturally at its peak. For weekly promotion scheduling, prioritize Saturday and Sunday, which consistently deliver 33–63% higher average sales than weekdays.

**2. Use SARIMAX as a Planning Tool, Not a Sole Decision-Maker**
Use the model to simulate promotion impact before launch and to support inventory planning. Treat forecasts as a benchmark for initial planning, to be combined with real-world conditions and team judgment.

**3. Apply Category-Specific Promotion Strategies**
For low-volume categories (e.g. Magazines, Books), consider different tactics like bundling with already-popular, high-selling categories. For high-volume categories (e.g. Groceries), the priority strategy is maintaining stock availability given consistently high demand.

---

## 🚀 Future Development

1. **Category-level models** — build separate SARIMAX models for priority categories (GROCERY I, BEVERAGES, PRODUCE) to get more precise, category-specific promotion coefficients.
2. **Additional exogenous variables** — incorporate Ecuadorian national holidays, pricing data, and macroeconomic indicators to improve model accuracy.
3. **Periodic model refresh** — the model trained on 2013–2016 data should be revalidated with more recent data, since consumer behavior patterns can shift over time.
4. **Explore alternative methods** — compare against LSTM or XGBoost with time series features to test whether a more complex model offers meaningfully better accuracy.

---

## 📂 Dataset Information

| Attribute | Detail |
|-----------|--------|
| **Dataset** | Favorita Store Sales (train.csv) |
| **Source** | [Kaggle — Store Sales: Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) |
| **Rows** | 3,000,888 |
| **Columns** | 6 |
| **Tools** | Python (Pandas, Statsmodels, pmdarima, Seaborn) + Tableau |

---

## 👤 Author

**Benedictus Irvanda Nugroho**
Data Analytics Portfolio Project · 2026

Focused on transforming raw business data into actionable insights through systematic data cleaning, rigorous exploratory analysis, and business intelligence reporting.

* **LinkedIn:** [irvandanugroho](https://linkedin.com/in/irvandanugroho/)
* **GitHub:** [Irvanda08](https://github.com/Irvanda08)
* **Email:** irvandanugroho08@gmail.com

---

*This project was completed as part of a professional Data Analytics portfolio (2026).*
