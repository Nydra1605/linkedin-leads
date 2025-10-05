import requests

payload = {
    "data":{
    "q": "openai",
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
    'authorization': "Bearer j1f2DhSUDBVeAfJINIlKDe63x4cPH1R8JnsJ9mFDf9PFVt1mTUBwadhUAdjC"
}

response = requests.post(url, headers=headers, json=payload)

print("Status Code:", response.status_code)
print("\nResponse JSON:")
print(response.json())
