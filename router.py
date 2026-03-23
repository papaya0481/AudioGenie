import yaml
from llm import (
    GeminiLLM, OpenaiLLM, NvidiaLLM, HuggingfaceLLM
)


def load_llm(name: str):
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    llm_config = config["llms"].get(name)
    if not llm_config:
        raise ValueError(f"LLM '{name}' not found in config.yaml")

    provider = llm_config["provider"]
    api_key = llm_config.get("api_key")
    model = llm_config.get("default_model")

    if provider == "openai":
        return OpenaiLLM(model=model, api_key=api_key, base_url=llm_config.get("api_url"))
    elif provider == "google":
        return GeminiLLM(model=model, api_key=api_key)
    elif provider == "nvidia":
        return NvidiaLLM(model=model, api_key=api_key, base_url=llm_config.get("api_url"))
    elif provider == "huggingface":
        return HuggingfaceLLM(model=model, **llm_config.get("parameters", {}))
    else:
        raise ValueError(f"Unsupported provider: {provider}")
