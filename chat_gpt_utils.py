from openai import OpenAI

def calculate_cost(completion_tokens, prompt_tokens, model_name):
   # gpt-3.5-turbo-1106	input $0.0010 / 1K tokens completion $0.0020 / 1K tokens
   if model_name == "gpt-3.5-turbo-0125":
        cost = completion_tokens / 1000 * 0.0015 + prompt_tokens / 1000 * 0.0005 
   elif model_name == 'gpt-3.5-turbo-instruct':
        cost = completion_tokens / 1000 * 0.002 + prompt_tokens / 1000 * 0.0015
   return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost}

def chat_completion(client, model_name, messages, seed, max_tokens, n, temperature = None, response_format = None):
    result = client.chat.completions.create(
                                model=model_name,
                                messages= messages,
                                max_tokens=max_tokens,
                                n=n,
                                seed = seed, logprobs=True,
                                temperature = temperature,
                                response_format = response_format
                                #stop=['\n\n'], # default is none
                                )
    return result

def initialize_openai_client(openai_key):
    client = OpenAI(api_key=openai_key)  # this is also the default, it can be omitted
    
    return client