import os
import pandas as pd
import requests
from openai import AzureOpenAI

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-05-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

# Load enriched company data
input_file = "test_augmented.csv"  # This file should already have AI_enabled, Sector, Funding, Links
output_file = "top_100_companies.csv"

df = pd.read_csv(input_file)

# Step 1: Prepare the dataset into JSON for LLM processing
company_records = df.to_dict(orient="records")

# Step 2: First LLM call was already done (data enrichment). Now we do the ranking & filtering.

prompt = f"""
You are given a list of companies with details:
- AI_enabled (Yes/No)
- Sector (e.g., Biotechnology, Pharmaceuticals, Biopharmaceuticals, etc.)
- Funding (Seed, Series A, Series B, etc.)

Rank and filter the companies to select the TOP 100 based on the following weighted criteria:
1. Prioritize companies that are NON-AI enabled (highest weight).
2. Among those, prioritize Biotechnology sector (medium weight).
3. Among those, prioritize companies with Series A or further funding (lowest weight).

Return the final top 100 companies as a structured JSON list with fields:
[Company Name, Website, AI_enabled, Sector, Funding, Link]
"""

response = client.chat.completions.create(
    model="GPT4o",
    messages=[
        {"role": "system", "content": "You are a helpful data ranking assistant."},
        {"role": "user", "content": prompt},
        {"role": "user", "content": str(company_records)}
    ],
    temperature=0
)

# Extract response
ranked_companies = response.choices[0].message.content

# Step 3: Parse JSON output
try:
    ranked_df = pd.read_json(ranked_companies)
except Exception as e:
    print("Error parsing JSON from LLM response:", e)
    ranked_df = pd.DataFrame()

# Step 4: Save top 100 companies to CSV
ranked_df.to_csv(output_file, index=False)
print(f"Top 100 companies saved to {output_file}")