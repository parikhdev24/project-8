# ==============================================================================
# JOB MARKET ANALYSIS AND RECOMMENDATION SYSTEM (TASKS 1 TO 8)
# Dataset: all_upwork_jobs_2024-02-07-2024-03-24.csv
# ==============================================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from scipy.stats import spearmanr, mannwhitneyu
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------------------------------------------------------------------
# DATA LOADING & PREPROCESSING
# ------------------------------------------------------------------------------
DATA_PATH = "all_upwork_jobs_2024-02-07-2024-03-24.csv"

df = pd.read_csv(DATA_PATH, low_memory=False)
df["title_clean"] = df["title"].fillna("").astype(str).str.strip()
df["title_lower"] = df["title_clean"].str.lower()
df["pub_dt"] = pd.to_datetime(df["published_date"], errors="coerce", utc=True)
df["pub_date"] = df["pub_dt"].dt.date
df["year_month"] = df["pub_dt"].dt.to_period("M").astype(str)
df["year_week"] = df["pub_dt"].dt.to_period("W").astype(str)

# Compute hourly midpoint & filter extreme outliers for clean rate analysis
df["hourly_mid"] = df[["hourly_low", "hourly_high"]].mean(axis=1)
hourly_df = df[df["hourly_mid"].between(3, 300)].copy()
budget_df = df[df["budget"].between(5, 50000)].copy()

# ==============================================================================
# TASK 1: ANALYZE THE CORRELATION BETWEEN JOB TITLE KEYWORDS AND OFFERED SALARIES
# ==============================================================================
print("\n=== TASK 1: JOB TITLE KEYWORDS VS. OFFERED SALARIES ===")

keywords = [
    "llm", "machine learning", "senior", "copywriter", "ai", "react",
    "python", "full stack", "figma", "wordpress", "video", "shopify",
    "seo", "data", "excel", "lead"
]

kw_results = []
for kw in keywords:
    pattern = r"\b" + kw.replace(" ", r"\s+") + r"\b"
    mask = df["title_lower"].str.contains(pattern, regex=True)

    h_with = df.loc[mask & df["hourly_mid"].between(3, 300), "hourly_mid"]
    h_without = df.loc[(~mask) & df["hourly_mid"].between(3, 300), "hourly_mid"]
    b_with = df.loc[mask & df["budget"].between(5, 50000), "budget"]

    # Mann-Whitney U test comparing hourly rates with vs. without keyword
    _, p_val = mannwhitneyu(h_with, h_without, alternative="two-sided") if len(h_with) > 10 else (np.nan, np.nan)

    kw_results.append({
        "keyword": kw,
        "total_postings": int(mask.sum()),
        "hourly_median": round(float(h_with.median()), 2),
        "hourly_mean": round(float(h_with.mean()), 2),
        "budget_median": round(float(b_with.median()), 2),
        "p_value": round(float(p_val), 6)
    })

kw_summary = pd.DataFrame(kw_results).sort_values("hourly_median", ascending=False)
print(kw_summary.to_string(index=False))

# Ridge regression on TF-IDF n-grams to quantify salary impact across all keywords
tfidf_sal = TfidfVectorizer(ngram_range=(1, 2), min_df=20, max_features=5000, stop_words="english")
X_sal = tfidf_sal.fit_transform(hourly_df["title_clean"])
y_sal = np.log1p(hourly_df["hourly_mid"])

ridge = Ridge(alpha=1.0)
ridge.fit(X_sal, y_sal)
feature_names = np.array(tfidf_sal.get_feature_names_out())
top_pos_idx = np.argsort(ridge.coef_)[-10:][::-1]
print("\nTop Salary-Boosting Title N-Grams (Ridge Coefficients):", list(zip(feature_names[top_pos_idx], np.round(ridge.coef_[top_pos_idx], 3))))

# Visualization for Task 1
plt.figure(figsize=(10, 6))
sns.barplot(data=kw_summary, x="hourly_median", y="keyword", palette="Blues_d")
plt.title("Task 1: Median Hourly Rate ($/hr) by Job Title Keyword")
plt.xlabel("Median Hourly Midpoint ($/hr)")
plt.ylabel("Keyword")
plt.tight_layout()
plt.savefig("task1_keyword_salaries.png", dpi=150)
plt.close()


# ==============================================================================
# TASK 2: IDENTIFY EMERGING JOB CATEGORIES BASED ON POSTING FREQUENCY
# ==============================================================================
print("\n=== TASK 2: EMERGING JOB CATEGORIES BY POSTING FREQUENCY ===")

categories = {
    "AI & GenAI": r"\b(ai|artificial intelligence|llm|gpt|openai|langchain|machine learning|ml)\b",
    "Web & App Dev": r"\b(react|node|full stack|frontend|backend|web developer|next\.?js|bubble|shopify|wordpress)\b",
    "Data & Analytics": r"\b(data analyst|data scientist|sql|power bi|tableau|data engineer|etl)\b",
    "Design & Creative": r"\b(figma|ui/ux|graphic design|logo|video editor|after effects|animation|illustrator)\b",
    "Marketing & Growth": r"\b(seo|ppc|facebook ads|google ads|social media|smma|lead generation|copywriter|content writer)\b",
    "Admin & Virtual Asst": r"\b(virtual assistant|data entry|customer support|bookkeeping|admin|excel)\b"
}

cat_monthly = pd.DataFrame({"year_month": sorted(df["year_month"].unique())})
for cat_name, regex_pat in categories.items():
    m = df["title_lower"].str.contains(regex_pat, regex=True)
    counts = df.loc[m].groupby("year_month").size()
    cat_monthly[cat_name] = cat_monthly["year_month"].map(counts).fillna(0).astype(int)

# Calculate Month-over-Month Growth from Feb 2024 to Mar 2024
feb_row = cat_monthly[cat_monthly["year_month"] == "2024-02"].iloc[0]
mar_row = cat_monthly[cat_monthly["year_month"] == "2024-03"].iloc[0]

growth_rows = []
for cat_name in categories:
    feb_cnt = int(feb_row[cat_name])
    mar_cnt = int(mar_row[cat_name])
    mom_pct = ((mar_cnt - feb_cnt) / feb_cnt * 100) if feb_cnt > 0 else np.nan
    growth_rows.append({
        "category": cat_name,
        "feb_2024": feb_cnt,
        "mar_2024": mar_cnt,
        "mom_growth_pct": round(mom_pct, 2)
    })

growth_df = pd.DataFrame(growth_rows).sort_values("mom_growth_pct", ascending=False)
print(growth_df.to_string(index=False))


# ==============================================================================
# TASK 3: PREDICT HIGH-DEMAND JOB ROLES BY ANALYZING POSTING PATTERNS OVER TIME
# ==============================================================================
print("\n=== TASK 3: PREDICT HIGH-DEMAND JOB ROLES & POSTING VOLUME ===")

daily_demand = df[df["pub_dt"] >= "2024-02-07"].groupby("pub_date").size().reset_index(name="postings")
daily_demand["pub_date"] = pd.to_datetime(daily_demand["pub_date"])
daily_demand = daily_demand.sort_values("pub_date").reset_index(drop=True)

# Time-series lag & calendar features
daily_demand["day_of_week"] = daily_demand["pub_date"].dt.dayofweek
daily_demand["lag_1"] = daily_demand["postings"].shift(1)
daily_demand["lag_7"] = daily_demand["postings"].shift(7)
daily_demand["rolling_7_mean"] = daily_demand["postings"].rolling(7).mean().shift(1)
ts_df = daily_demand.dropna().copy()

split = int(len(ts_df) * 0.8)
features = ["day_of_week", "lag_1", "lag_7", "rolling_7_mean"]
X_train, X_test = ts_df[features].iloc[:split], ts_df[features].iloc[split:]
y_train, y_test = ts_df["postings"].iloc[:split], ts_df["postings"].iloc[split:]

rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"Forecast Accuracy -> MAE: {mae:.2f} | RMSE: {rmse:.2f} | R2: {r2:.4f}")

# Visualization for Task 3
plt.figure(figsize=(10, 5))
plt.plot(ts_df["pub_date"].iloc[split:], y_test.values, label="Actual Postings", marker="o")
plt.plot(ts_df["pub_date"].iloc[split:], y_pred, label="Predicted Postings (RF)", marker="x")
plt.title(f"Task 3: Daily Job Demand Forecast (MAE={mae:.1f}, R2={r2:.2f})")
plt.xlabel("Date")
plt.ylabel("Daily Job Postings")
plt.xticks(rotation=30)
plt.legend()
plt.tight_layout()
plt.savefig("task3_demand_forecast.png", dpi=150)
plt.close()


# ==============================================================================
# TASK 4: COMPARE AVERAGE HOURLY RATES ACROSS DIFFERENT COUNTRIES
# ==============================================================================
print("\n=== TASK 4: AVERAGE HOURLY RATES ACROSS COUNTRIES ===")

country_rates = (
    hourly_df.groupby("country")["hourly_mid"]
    .agg(postings="count", median_hourly="median", mean_hourly="mean")
    .query("postings >= 300")
    .sort_values("median_hourly", ascending=False)
    .round(2)
    .reset_index()
)
print(country_rates.head(15).to_string(index=False))

# Interactive Plotly Choropleth Map
fig_map = px.choropleth(
    country_rates,
    locations="country",
    locationmode="country names",
    color="median_hourly",
    hover_name="country",
    hover_data={"postings": True, "mean_hourly": True, "median_hourly": True},
    color_continuous_scale="Teal",
    title="Task 4: Median Hourly Rate ($/hr) by Country (Min 300 Postings)"
)
fig_map.write_html("task4_country_hourly_map.html")
print("Saved interactive map -> task4_country_hourly_map.html")


# ==============================================================================
# TASK 5: CREATE A JOB RECOMMENDATION ENGINE BASED ON CURRENT JOB POSTINGS
# ==============================================================================
print("\n=== TASK 5: PERSONALIZED JOB RECOMMENDATION ENGINE ===")

class JobRecommender:
    def __init__(self, jobs_df: pd.DataFrame):
        self.jobs = jobs_df.reset_index(drop=True).copy()
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=20000)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.jobs["title_clean"])

    def recommend(self, query: str, country: str = None, min_hourly: float = None, top_k: int = 5) -> pd.DataFrame:
        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.tfidf_matrix).flatten()

        candidates = self.jobs.copy()
        candidates["match_score"] = np.round(scores, 4)

        if country:
            candidates = candidates[candidates["country"].fillna("").str.lower() == country.lower()]
        if min_hourly is not None:
            candidates = candidates[candidates["hourly_mid"] >= min_hourly]

        cols = ["title_clean", "country", "is_hourly", "hourly_mid", "budget", "match_score", "link"]
        return candidates.sort_values("match_score", ascending=False)[cols].head(top_k)

recommender = JobRecommender(df)
sample_recs = recommender.recommend(
    query="python machine learning llm api developer",
    country="United States",
    min_hourly=30.0,
    top_k=5
)
print(sample_recs.to_string(index=False))


# ==============================================================================
# TASK 6: TRACK CHANGES IN JOB MARKET DYNAMICS OVER MONTHS
# ==============================================================================
print("\n=== TASK 6: MONTHLY JOB MARKET DYNAMICS TRACKER ===")

monthly_dynamics = (
    df[df["year_month"].isin(["2024-02", "2024-03"])]
    .groupby("year_month")
    .agg(
        total_postings=("title_clean", "count"),
        hourly_share_pct=("is_hourly", lambda s: round(s.mean() * 100, 2)),
        median_hourly_rate=("hourly_mid", lambda s: round(s[s.between(3, 300)].median(), 2)),
        median_fixed_budget=("budget", lambda s: round(s[s.between(5, 50000)].median(), 2)),
        active_countries=("country", "nunique")
    )
    .reset_index()
)
print(monthly_dynamics.to_string(index=False))


# ==============================================================================
# TASK 7: INVESTIGATE TRENDS IN THE REMOTE WORK LANDSCAPE
# ==============================================================================
print("\n=== TASK 7: REMOTE WORK & CONTRACT FLEXIBILITY TRENDS ===")

remote_regex = r"\b(remote|work from home|wfh|async|asynchronous|flexible|part-?time)\b"
df["explicit_remote_keyword"] = df["title_lower"].str.contains(remote_regex, regex=True)

remote_summary = (
    df[df["year_month"].isin(["2024-02", "2024-03"])]
    .groupby("year_month")
    .agg(
        total_jobs=("title_clean", "count"),
        explicit_remote_mentions=("explicit_remote_keyword", "sum"),
        explicit_remote_pct=("explicit_remote_keyword", lambda s: round(s.mean() * 100, 3)),
        hourly_contract_pct=("is_hourly", lambda s: round(s.mean() * 100, 2))
    )
    .reset_index()
)
print(remote_summary.to_string(index=False))


# ==============================================================================
# TASK 8: PREDICT FUTURE JOB MARKET TRENDS (SCENARIO MODELING)
# ==============================================================================
print("\n=== TASK 8: FUTURE JOB MARKET SCENARIO PROJECTIONS (6-MONTH HORIZON) ===")

mar_total = int((df["year_month"] == "2024-03").sum())
mar_ai = int(mar_row["AI & GenAI"])
mar_web = int(mar_row["Web & App Dev"])

scenarios = pd.DataFrame([
    {
        "Scenario": "1. AI-Accelerated Automation",
        "Monthly_AI_Growth_Assumption": "+18% MoM",
        "Projected_AI_Jobs_Month_6": int(mar_ai * (1.18 ** 6)),
        "Projected_Web_Jobs_Month_6": int(mar_web * (1.06 ** 6)),
        "Expected_Rate_Shift": "+12% for AI/LLM, -5% for routine tasks"
    },
    {
        "Scenario": "2. Steady Digital Expansion",
        "Monthly_AI_Growth_Assumption": "+8% MoM",
        "Projected_AI_Jobs_Month_6": int(mar_ai * (1.08 ** 6)),
        "Projected_Web_Jobs_Month_6": int(mar_web * (1.08 ** 6)),
        "Expected_Rate_Shift": "+3% across core tech & creative roles"
    },
    {
        "Scenario": "3. Budget-Constrained Efficiency",
        "Monthly_AI_Growth_Assumption": "+5% MoM",
        "Projected_AI_Jobs_Month_6": int(mar_ai * (1.05 ** 6)),
        "Projected_Web_Jobs_Month_6": int(mar_web * (1.02 ** 6)),
        "Expected_Rate_Shift": "Shift toward fixed-price milestones & global hubs"
    }
])
print(scenarios.to_string(index=False))

months = [f"M+{i}" for i in range(0, 7)]
ai_s1 = [int(round(mar_ai * (1.18**i))) for i in range(7)]
ai_s2 = [int(round(mar_ai * (1.08**i))) for i in range(7)]
ai_s3 = [int(round(mar_ai * (1.05**i))) for i in range(7)]

web_s1 = [int(round(mar_web * (1.06**i))) for i in range(7)]
web_s2 = [int(round(mar_web * (1.08**i))) for i in range(7)]
web_s3 = [int(round(mar_web * (1.02**i))) for i in range(7)]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(months, ai_s1, marker="o", label="1. AI-Accelerated (+18%/mo)")
axes[0].plot(months, ai_s2, marker="s", label="2. Steady Digital (+8%/mo)")
axes[0].plot(months, ai_s3, marker="^", label="3. Budget-Constrained (+5%/mo)")
axes[0].set_title(
    "Task 8: Projected AI & GenAI Job Postings (6-Month Horizon)"
)
axes[0].set_xlabel("Projection Horizon (Months from Mar 2024 Baseline)")
axes[0].set_ylabel("Projected Monthly Postings")
axes[0].legend(fontsize=8)

axes[1].plot(months, web_s1, marker="o", label="1. AI-Accelerated (+6%/mo)")
axes[1].plot(months, web_s2, marker="s", label="2. Steady Digital (+8%/mo)")
axes[1].plot(months, web_s3, marker="^", label="3. Budget-Constrained (+2%/mo)")
axes[1].set_title(
    "Task 8: Projected Web & App Dev Job Postings (6-Month Horizon)"
)
axes[1].set_xlabel("Projection Horizon (Months from Mar 2024 Baseline)")
axes[1].set_ylabel("Projected Monthly Postings")
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig("task8_future_scenarios.png", dpi=150)
plt.close()