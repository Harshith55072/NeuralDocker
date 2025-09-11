import requests

url = "http://127.0.0.1:8000/chat"   # FastAPI server URL

payload = {
    "user_input": "hi",
    "role": "Career_mentor"
}

res = requests.post(url, json=payload)

if res.status_code == 200:
    print("✅ Response:", res.json()["response"])
else:
    print("❌ Error:", res.status_code, res.text)
