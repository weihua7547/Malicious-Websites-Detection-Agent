import requests

def test_ollama():
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llama3.1:8b",
        "prompt": "Hello! Are you ready to help me classify URLs?",
        "stream": False
    }
    
    try:
        response = requests.post(url, json=payload)
        print("連線成功！模型回覆：")
        print(response.json()['response'])
    except Exception as e:
        print(f"連線失敗，請確認 Ollama 是否正在運行。錯誤：{e}")

if __name__ == "__main__":
    test_ollama()