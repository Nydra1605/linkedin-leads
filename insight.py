import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

# === Load Data ===
file_path = "serp_output.xlsx"
df = pd.read_excel(file_path)

# === Clean Data ===
df = df.drop_duplicates()
df.columns = df.columns.str.lower().str.strip()

# Replace "Not Found" with NaN
df = df.replace("Not Found", pd.NA)

# === Funding Normalization ===
def parse_funding(value):
    """Convert funding text (e.g., '$80M Series C', '$2.5B') to numeric (in million USD)."""
    if pd.isna(value):
        return None
    match = re.search(r'\$([\d\.]+)\s*([MB]?)', value)
    if match:
        amount = float(match.group(1))
        unit = match.group(2).upper()
        if unit == 'B':
            return amount * 1000  # Convert billions to millions
        elif unit == 'M':
            return amount
        else:
            return amount / 1_000_000  # Assume raw USD value
    return None

df['funding_usd_m'] = df['funding'].apply(parse_funding)

# === 1️⃣ Overview ===
print("🔹 Total companies:", len(df))
print("🔹 Total sectors:", df['sector'].nunique())
print("🔹 AI-enabled breakdown:\n", df['ai_enabled'].value_counts(), "\n")

# === 2️⃣ Companies by Sector ===
sector_counts = df['sector'].value_counts()
plt.figure(figsize=(10,6))
sns.barplot(x=sector_counts.values, y=sector_counts.index, palette='viridis')
plt.title("Number of Companies by Sector")
plt.xlabel("Number of Companies")
plt.ylabel("Sector")
plt.tight_layout()
plt.show()

# === 3️⃣ AI-enabled Breakdown ===
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='ai_enabled', palette='coolwarm')
plt.title("AI-Enabled vs Non-AI Companies")
plt.xlabel("AI Enabled?")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# === 4️⃣ Funding Insights ===
funded_companies = df[df['funding_usd_m'].notna()]
print(f"💰 {len(funded_companies)} companies have valid funding data.")

# Distribution plot
plt.figure(figsize=(8,5))
sns.histplot(funded_companies['funding_usd_m'], bins=20, kde=True)
plt.title("Funding Distribution (in Million USD)")
plt.xlabel("Funding (Million USD)")
plt.ylabel("Number of Companies")
plt.tight_layout()
plt.show()

# Top 10 funded companies
top_funded = funded_companies.sort_values(by='funding_usd_m', ascending=False).head(10)
plt.figure(figsize=(10,5))
sns.barplot(x='funding_usd_m', y='company_name', data=top_funded, palette='magma')
plt.title("Top 10 Companies by Funding (in Million USD)")
plt.xlabel("Funding (Million USD)")
plt.ylabel("Company")
plt.tight_layout()
plt.show()

# === 5️⃣ Sector-wise Funding Summary ===
sector_summary = (
    df.groupby('sector', dropna=False)
      .agg(
          total_companies=('company_name', 'count'),
          funded_companies=('funding_usd_m', lambda x: x.notna().sum()),
          avg_funding_musd=('funding_usd_m', 'mean'),
          ai_enabled_companies=('ai_enabled', lambda x: (x == 'Yes').sum())
      )
      .reset_index()
      .sort_values(by='total_companies', ascending=False)
)

# === 6️⃣ AI-enabled Summary ===
ai_summary = (
    df.groupby('ai_enabled', dropna=False)
      .agg(
          total_companies=('company_name', 'count'),
          funded_companies=('funding_usd_m', lambda x: x.notna().sum()),
          avg_funding_musd=('funding_usd_m', 'mean')
      )
      .reset_index()
)

# === 7️⃣ Save Report ===
with pd.ExcelWriter("company_insights_report.xlsx") as writer:
    df.to_excel(writer, index=False, sheet_name="Raw Data (Cleaned)")
    sector_summary.to_excel(writer, index=False, sheet_name="Sector Summary")
    ai_summary.to_excel(writer, index=False, sheet_name="AI Summary")
    top_funded.to_excel(writer, index=False, sheet_name="Top Funded Companies")

print("✅ Saved detailed report as 'company_insights_report.xlsx'")

# === 8️⃣ Additional Insight ===
# Correlation between AI-enablement and funding presence
ai_funding_corr = pd.crosstab(df['ai_enabled'], df['funding_usd_m'].notna(), normalize='index') * 100
print("\n📈 Percentage of companies with funding (by AI status):")
print(ai_funding_corr)
