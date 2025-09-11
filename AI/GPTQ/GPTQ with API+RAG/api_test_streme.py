import requests

url = "http://127.0.0.1:8000/chat/stream"   # <-- FIXED

payload = {
    "user_input": "hi",
    "role": "Career_mentor"
}

with requests.post(url, json=payload, stream=True) as res:
    if res.status_code == 200:
        print("✅ Streaming response:")
        for chunk in res.iter_content(chunk_size=None, decode_unicode=True):
            if chunk:
                print(chunk, end="", flush=True)
    else:
        print("❌ Error:", res.status_code, res.text)
