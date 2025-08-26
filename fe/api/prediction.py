import requests
import json
import os

from typing import Dict
from dotenv import load_dotenv

load_dotenv(dotenv_path=".dev.env")

URL=os.environ.get('BE_APP_HOST')
PORT=os.getenv('BE_APP_PORT')

def get_pred(data: Dict):
    req = requests.post(url=f"http://{URL}:{PORT}/predict", json=data)
    result = req.json()

    message = result.get("message", "")
    result = result.get("result", "")
    return message, result