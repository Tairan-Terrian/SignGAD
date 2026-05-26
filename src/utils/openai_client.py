from openai import OpenAI
from utils.openai_config import apply_openai_env, openai_client_kwargs, selector_model

apply_openai_env()


def query_openai(messages, model=None):
    client = OpenAI(**openai_client_kwargs())
    response = client.chat.completions.create(
        model=model or selector_model(),
        messages=messages
    )
    return response.choices[0].message.content
