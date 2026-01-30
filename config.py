import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    TENK_DATA_EXTRACTOR_OPENAI_MODEL_NAME = os.getenv("TENK_DATA_EXTRACTOR_OPENAI_MODEL_NAME", "gpt-3.5-turbo")

def get_config():
    return Config()