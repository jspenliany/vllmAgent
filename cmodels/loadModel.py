from langchain_openai import ChatOpenAI
from pydantic import SecretStr
import logging

log=logging.getLogger("chatAsYou260325")
# Use a global variable to act as a cache
_llm_instance = None

def load_local_model():
    global _llm_instance
    if _llm_instance is None:
        log.debug("--- Initializing Gemma-3-27B (First Time) ---")
        # Your actual loading logic here
        _llm_instance = ChatOpenAI(
            model="gemma-3-27b-it",
            base_url="http://0.0.0.0:8000/v1",
            api_key=SecretStr("empty")
        )
    return _llm_instance
