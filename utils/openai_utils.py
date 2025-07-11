import time
import json
import os

import tiktoken
import openai
from openai import OpenAI


## example code from OpenAI to calculate the number of tokens in messages
def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    num_tokens = 0

    if model.startswith("gpt-"):
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
    elif model.startswith("text-davinci") or model.startswith("davinci-codex"):
        for message in messages:
            num_tokens += len(encoding.encode(message.get("role", ""))) + len(encoding.encode(message.get("content", "")))
            if "name" in message:
                num_tokens += len(encoding.encode(message.get("name", "")))
    else:
        raise NotImplementedError(f"num_tokens_from_messages() is not presently implemented for model {model}.")

    return num_tokens


def init_gpt(OPENAI_API_KEY, ORGANIZATION_ID):
    """Initialize OpenAI API with API key and organization ID"""
    openai.api_key = OPENAI_API_KEY
    openai.organization = ORGANIZATION_ID
    return


def init_openai_client(openai_api_key:str = None, openai_organization:str = None):
    """Initialize OpenAI client with credentials, timeout and max_retries"""
    if openai_api_key is None:
        openai_api_key = os.getenv('OPENAI_API_KEY')
    if openai_organization is None:
        openai_organization = os.getenv('OPENAI_ORGANIZATION')
    client = OpenAI(
        api_key=openai_api_key,
        organization=openai_organization,
        timeout=180.0,
        max_retries=5,
    )
    return client


def ask_gpt(client, model, messages: list, temperature, n, seed=None):
    """Call API to get response from model
    """
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        n=n,
        seed=seed,
    )
    response_clean = [choice.message.content for choice in response.choices]
    if n == 1:
        response_clean = response_clean[0]
    return dict(
        response=response_clean,
        **response.usage.__dict__
    )


def get_prompt_from_openai(client:None, model:str, data: str|dict, temperature: float, n:int, seed=None, max_num_retry=5, flag_use_original=False, flag_return_text_only=False):
    """Get prompt from OpenAI API
    """
    if client is None:
        client = init_openai_client() 
    num_retry = 0
    response = None
    while num_retry < max_num_retry:
        try:
            if isinstance(data, str):
                messages = [{"role": "user", "content": data}]
            else:
                messages = data
            if flag_use_original:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    n=n,
                    seed=seed,
                )
                if flag_return_text_only:
                    if n == 1:
                        response = response.choices[0].message.content
                    else:
                        response = [choice.message.content for choice in response.choices]
            else:
                ## transaction in text2sql
                response = ask_gpt(client, model, messages, temperature, n, seed)
                response['response'] = [response['response']]
                if flag_return_text_only:
                    response = response['response']
            break
        except openai.APIConnectionError as e:
            print("The server could not be reached")
            print(e.__cause__)  # an underlying Exception, likely raised within httpx.
            num_retry += 1
            time.sleep(1)
        except openai.RateLimitError as e:
            print("A 429 status code was received; we should back off a bit.")
            num_retry += 1
            time.sleep(1)
        except openai.APIStatusError as e:
            print("Another non-200-range status code was received")
            print(e.status_code)
            print(e.response)
            num_retry += 1
            time.sleep(1)
        except json.decoder.JSONDecodeError:
            # json decoder error doesn't need to retry
            print(f"JSONDecodeError", end="\n")
            return None
        except Exception as e:
            num_retry += 1
            print(f"Repeat for the {num_retry} times for exception: {e}", end="\n")
            time.sleep(1)
    else:
        # If we exhaust max_num_retry, log that outcome
        print("Failed to get a response after maximum retries.")
        return None
    return response

def get_price_from_tokens(num_tokens:int, model:str):
    costs_per_thousand = {
        'gpt-4': 0.03,
        'gpt-3.5-turbo-0125': 0.0005,
    }
    if model in costs_per_thousand:
        cost_per_thousand = costs_per_thousand[model]
        return num_tokens * cost_per_thousand / 1000
    else:
        return None
    