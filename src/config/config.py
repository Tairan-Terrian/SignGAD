import os


class Config:
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or "YOUR_API_KEY"
    OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE") or "YOUR_API_BASE_URL"
    OPENAI_MODEL = os.environ.get("OPENAI_MODEL") or "gpt-4.1"
    SELECTOR_MODEL = os.environ.get("ADAGENT_SELECTOR_MODEL") or OPENAI_MODEL

    @classmethod
    def apply_openai_env(cls) -> None:
        os.environ["OPENAI_API_KEY"] = cls.OPENAI_API_KEY
        os.environ["OPENAI_MODEL"] = cls.OPENAI_MODEL
        if cls.OPENAI_API_BASE:
            os.environ["OPENAI_API_BASE"] = cls.OPENAI_API_BASE

    @classmethod
    def openai_client_kwargs(cls) -> dict:
        kwargs = {"api_key": cls.OPENAI_API_KEY}
        if cls.OPENAI_API_BASE:
            kwargs["base_url"] = cls.OPENAI_API_BASE
        return kwargs

    @classmethod
    def chat_openai_kwargs(cls, temperature: float = 0) -> dict:
        kwargs = {
            "model": cls.OPENAI_MODEL,
            "temperature": temperature,
            "api_key": cls.OPENAI_API_KEY,
        }
        if cls.OPENAI_API_BASE:
            kwargs["base_url"] = cls.OPENAI_API_BASE
        return kwargs
