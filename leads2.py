# from langchain_community.utilities import GoogleSerperAPIWrapper
# from dotenv import load_dotenv
# import os
# load_dotenv()
# SERPER_API_KEY =os.getenv("SERPER_API_KEY")
# search = GoogleSerperAPIWrapper()
# output = search.run("Obama's first name?")
# print(output)

from langchain.chat_models import init_chat_model
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.tools import Tool
from langgraph.prebuilt import create_react_agent
import os
from dotenv import load_dotenv
import pandas as pd
import json

load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
DEPLOYMENT_NAME = "GPT4o" 
AZURE_OPENAI_API_VERSION = "2023-07-01-preview"
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
os.environ["OPENAI_API_VERSION"] = "2023-07-01-preview"

llm = init_chat_model("GPT4o", model_provider="azure_openai", temperature=0)

search = GoogleSerperAPIWrapper()

def search_with_links(query: str):
    """Wrapper that returns both answer text and source links."""
    raw = search.results(query)  # returns JSON with results
    links = [item.get("link") for item in raw.get("organic", [])[:3] if "link" in item]  # top 3 links
    answer = "\n".join([item.get("snippet", "") for item in raw.get("organic", [])])
    return {"answer": answer, "links": links}

tools = [
    Tool(
        name="Intermediate_Answer",
        func=search.run,
        description="useful for when you need to ask with search",
    )
]
agent = create_react_agent(llm, tools)

# events = agent.stream(
#     {
#         "messages": [
#             ("user", "What is the hometown of the reigning men's U.S. Open champion?")
#         ]
#     },
#     stream_mode="values",
# )


# for event in events:
#     event["messages"][-1].pretty_print()

df_input = pd.read_csv("test.csv")   # assumes there's a "Company" column
companies = df_input['Lead_Company'].drop_duplicates().tolist()
    
# Remove any NaN values if they exist
companies = [company for company in companies if pd.notna(company)]
    
print(f"Loaded {len(companies)} unique companies from {'test.csv'}")



# -------------------------
# 🔍 Query Template
# -------------------------
query_template = """
Extract structured company information for {company}.
Return results ONLY in valid JSON with the following fields:

1. Sector
2. Funding
3. Location
4. AI Enabled (Yes/No)
5. CEO
6. CEO Linkedin
7. CSO
8. CSO Linkedin
9. Therapeutic Area
10. Molecule Focus
11. Clinical Stage
"""

# # -------------------------
# # 🚀 Run Agent per Company
# # -------------------------
# results = []
# for company in companies:
#     events = agent.stream(
#         {
#             "messages": [
#                 ("user", query_template.format(company=company))
#             ]
#         },
#         stream_mode="values",
#     )

#     final_message = None
#     for event in events:
#         if event["messages"][-1].type == "ai":
#             final_message = event["messages"][-1].content

#     if final_message:
#         try:
#             # Ensure JSON safety
#             data = json.loads(final_message.strip("```json").strip("```"))
#             data["Company"] = company
#             results.append(data)
#         except Exception:
#             results.append({"Company": company, "Error": final_message})

# # -------------------------
# # 📊 Save to CSV
# # -------------------------
# df_output = pd.DataFrame(results)
# df_output.to_csv("company_augmented_data.csv", index=False)
# print("✅ Augmented data saved to company_augmented_data.csv")

# -------------------------
# 🚀 Run Agent per Company
# -------------------------
results = []
for company in companies:
    # Run search first to grab funding links
    funding_search = search_with_links(f"{company} funding rounds investment venture capital")
    funding_links = "; ".join(funding_search["links"]) if funding_search["links"] else ""

    # Run agent for structured info
    events = agent.stream(
        {
            "messages": [
                ("user", query_template.format(company=company))
            ]
        },
        stream_mode="values",
    )

    final_message = None
    for event in events:
        if event["messages"][-1].type == "ai":
            final_message = event["messages"][-1].content

    if final_message:
        try:
            cleaned = final_message.strip().strip("```json").strip("```")
            data = json.loads(cleaned)
            data["Company"] = company
            data["Funding_Links"] = funding_links   # ✅ store funding links
            results.append(data)
        except Exception:
            results.append({"Company": company, "Error": final_message, "Funding_Links": funding_links})

# -------------------------
# 📊 Merge Back into Original CSV
# -------------------------
df_augmented = pd.DataFrame(results)

# Merge on "Company"
# df_final = df_input.merge(df_augmented, on="Company", how="left")

# Ensure required final columns
cols_order = (
    ["Company", "Company Website", "Sector", "Funding", "Funding_Links", "Location",
     "AI Enabled (Yes/No)", "CEO", "CEO Linkedin", "CSO", "CSO Linkedin",
     "Therapeutic Area", "Molecule Focus", "Clinical Stage"]
)
df_final = df_augmented[[c for c in cols_order if c in df_augmented.columns] + 
                    [c for c in df_augmented.columns if c not in cols_order]]

# Save
df_final.to_csv("test_augmented.csv", index=False)
print("✅ Augmented data merged and saved to test_augmented.csv")
