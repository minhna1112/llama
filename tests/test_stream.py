import requests
import json

url = "http://{APP_HOST}:{APP_PORT}/chat_completion_stream"

payload = json.dumps([
  {
    "role": "system",
    "content": "You are my coding assistant"
  },
  {
    "role": "user",
    "content": "Help me!"
  },
  {
    "role": "assistant",
    "content": "Okay bro!"
  },
  {
    "role": "user",
    "content": "Generate Python code for Djikstra algorithm "
  }
])
headers = {
  'Content-Type': 'application/json',
}

response = requests.request("GET", url, headers=headers, data=payload, stream=True)

for res in response:
    print(res)