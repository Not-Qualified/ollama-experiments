from pprint import pprint
from typing import Generator, Optional, Sequence

import ollama
from ollama._types import ListResponse

from helper.utils import cache_tmp


def get_all_models(filter_out=None) -> Generator[ListResponse.Model,None,None]:
    if filter_out is None:
        filter_out = list()
    filter_out.append('llama4:latest')
    for _ in ollama.list().models:
        if filter_out and any(kw in _.model for kw in filter_out):
            continue
        yield _


@cache_tmp
def get_models_capability(model_obj):
    return ollama.chat(
        model_obj.model,
        [{'content': "What can you do? describe in only keywords", 'role': 'user'}],
        options=ollama.Options(num_thread=15, )
    )['message']['content']


def choose_best_model_from_prompt(prompt: str) -> str | None:
    ask_to_ollama = '''which model is best for this prompt? following are the models and their capabilities
    '''

    for llm_model in get_all_models(filter_out=['embed']):
        ask_to_ollama += f"{llm_model.model} : {get_models_capability(llm_model)}\n"

    ask_to_ollama += '''Please choose the best model for this prompt, and provide the model name as the response.
    '''

    ask_to_ollama += prompt

    output = ollama.chat(
        'qwen3:4b',
        [ollama.Message(role='user', content=ask_to_ollama)],
        options=ollama.Options(num_thread=15)
    )

    pprint(output)

    for _ in get_all_models():
        if _.model.lower() in output.message.content.lower():
            return _.model
        return None
    return None


if __name__ == "__main__":
    for i, model in enumerate(get_all_models(filter_out=['embed'])):
        print(i, model.model)

