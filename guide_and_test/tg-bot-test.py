import requests
import os
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_test_message():
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": "🚀 交易机器人测试消息！一切正常。"
    }
    response = requests.post(url, json=payload)
    print(response.json())  # 如果成功，会打印 {"ok": true, ...}

if __name__ == "__main__":
    send_test_message()