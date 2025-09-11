import requests

url = "http://127.0.0.1:8000/chat/stream"

payload = {
    "user_input": "hi",
    "role": "Career_mentor"
}

with requests.post(url, json=payload, stream=True) as res:
    if res.status_code == 200:
        print("✅ Streaming response:\n")
        for line in res.iter_lines(decode_unicode=True):
            if not line:  # skip keep-alive newlines
                continue
            if line.startswith("data: "):
                data = line[len("data: "):]  # remove prefix
                if data == "[DONE]":
                    print("\n\n✅ Stream finished.")
                    break
                print(data, end="", flush=True)
    else:
        print("❌ Error:", res.status_code, res.text)
