from google import genai
from google.genai import types
from google.colab import userdata
import pandas as pd
import time

client = genai.Client(api_key=userdata.get('GEMINI_API_KEY'))

grounding_tool = types.Tool(
    google_search=types.GoogleSearch()
)

config = types.GenerateContentConfig(
    tools=[grounding_tool]
)
df = pd.read_csv("serp_test.csv")
for index, row in df.iterrows():
    company_name = row['Lead_Company']
    response1 = client.models.generate_content(
      model="gemini-2.5-flash",
      contents=f"{company_name} Sector, either Pharmaceuticals, Biotechnology or Others, give answer in 1-2 words only, don't add any other text, give N/A if you do not know the answer",
      config=config,
    )
    Sector = response1.text
    print(response1.text)
    df.at[index, "Sector"] = Sector
    time.sleep(3)

    response2 = client.models.generate_content(
      model="gemini-2.5-flash",
      contents=f"{company_name} latest funding, give only amount with stage of funding, don't add any other text",
      config=config,
    )
    Funding = response2.text
    df.at[index, "Funding"] = Funding

    data = None
    uri = None
    if response2.candidates and response2.candidates[0].grounding_metadata and response2.candidates[0].grounding_metadata.grounding_chunks:
      data = response2.candidates[0].grounding_metadata.grounding_chunks
      if data and data[0].web:
        uri = data[0].web.uri

    df.at[index, "Funding_Citation"] = uri
    time.sleep(3)

    response3 = client.models.generate_content(
      model="gemini-2.5-flash",
      contents=f"{company_name} AI Capabilities, give only Yes or No, give N/A if you do not know",
      config=config,
    )
    AI_Capabilities = response3.text
    df.at[index, "AI_Capabilities"] = AI_Capabilities

df.to_csv("gemini_output.csv", index=False)