import requests

url = "http://127.0.0.1:7700/verify"  # match the port in liveness_service.py
data = {
    "embedding1": [0.12, 0.34, 0.56, 0.78],
    "embedding2": [0.11, 0.35, 0.57, 0.79]
}

resp = requests.post(url, json=data)
print(resp.status_code)
print(resp.text)  # use text first to see raw response
