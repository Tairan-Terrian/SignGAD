import os

from config.config import Config


def apply_openai_env() -> None:
    api_key = getattr(Config, "OPENAI_API_KEY", "")
    api_base = getattr(Config, "OPENAI_API_BASE", "")
    model = getattr(Config, "OPENAI_MODEL", "gpt-4.1")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_MODEL"] = model
    if api_base:
        os.environ["OPENAI_API_BASE"] = api_base


def openai_client_kwargs() -> dict:
    kwargs = {}
    api_key = getattr(Config, "OPENAI_API_KEY", "")
    api_base = getattr(Config, "OPENAI_API_BASE", "")
    if api_key:
        kwargs["api_key"] = api_key
    if api_base:
        kwargs["base_url"] = api_base
    return kwargs


def chat_openai_kwargs(temperature: float = 0) -> dict:
    kwargs = {
        "model": getattr(Config, "OPENAI_MODEL", "gpt-4.1"),
        "temperature": temperature,
    }
    kwargs.update(openai_client_kwargs())
    return kwargs


def selector_model() -> str:
    return getattr(
        Config,
        "SELECTOR_MODEL",
        getattr(Config, "OPENAI_MODEL", "gpt-4.1"),
    )
