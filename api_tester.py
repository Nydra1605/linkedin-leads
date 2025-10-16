import requests
from dotenv import load_dotenv
import os
load_dotenv()

api_key = os.getenv("SERPHOUSE_API_KEY")

payload = {
    "data":{
    "q": "aurigene oncology latest project",
    "domain": "google.com",
    "loc": "United States",
    "lang": "en",
    "device": "desktop",
    "serp_type": "web",
    }
}

url = "https://api.serphouse.com/serp/live"
headers={
    'accept': 'application/json',
    'content-type': 'application/json',
    'authorization': f"Bearer {api_key}"
}

response = requests.post(url, headers=headers, json=payload)

print("Status Code:", response.status_code)
print("\nResponse JSON:")
print(response.json())
