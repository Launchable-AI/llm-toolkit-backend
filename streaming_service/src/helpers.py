from .log_config import logging
logger = logging.getLogger(__name__)

import json
import httpx
import traceback
from functools import lru_cache

from .models import FunctionCall, ResponseMessage, ChatCompletion
from .redis_manager import redis_manager
from .security import verify_token, decrypt_jwt

from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException

import tiktoken

# instatiate the redis manager
# TODO - add config file here if needing to connect

async def prepare_openai_messages(chat_completion):
    ''' Prepare the messages for the OpenAI API call.

    If last_n_messages is set, use only the last n messages.

    If smart_truncate is True, auto-truncate the messages.

    If first message is system message, add it back on to start of list.
    '''

    try:
        messages = chat_completion.messages

        if chat_completion.last_n_messages is not None:
            messages = chat_completion.messages[-chat_completion.last_n_messages:]

        if chat_completion.smart_truncate:
            messages = await auto_truncate_messages(
                messages, chat_completion.model, chat_completion.max_input_tokens)
        
        # Ensure that if the original first message is a system message, 
        # and it's not already at the start of the list, add it back.
        if (chat_completion.messages and 
            chat_completion.messages[0].get("role") == "system" and 
            messages[0].get("role") != "system"):
            messages.insert(0, chat_completion.messages[0])

        return messages
    except Exception as e:
        logger.error("Error preparing OpenAI messages:", exc_info=True)
        raise e

async def prepare_openai_request(chat_completion: ChatCompletion):
    '''
    Prepare the request for the OpenAI / OpenRouter API call.

    Note that API key selection has gotten complicated.  Could be:
    - OpenAI key (in token or in request)
    - OpenRouter key (in token or in request)
    - Any other key (eg Azure) in headers (in token or in request)
    '''
    try:

        # set these to None so we don't have an undeclared error
        token_headers = None
        api_key = None

        if chat_completion.custom_endpoint:
            url = chat_completion.custom_endpoint
        elif chat_completion.use_openrouter:
            url = 'https://openrouter.ai/api/v1/chat/completions'
        else:
            url = 'https://api.openai.com/v1/chat/completions'

        # Extract values from security token
        if chat_completion.security_token:
            
            # if token starts with security marker, remove it and decrypt
            if chat_completion.security_token and chat_completion.security_token.startswith("azx09ap29t"):
                decoded_token = await decrypt_jwt(chat_completion.security_token)
            else:
                decoded_token = chat_completion.security_token

            # veryify token
            valid_token = await verify_token(
                decoded_token,
                chat_completion.app_url
                )

            if valid_token is False:
                raise HTTPException(status_code=401, detail="Invalid token")

            openai_api_key = decoded_token.get("api_key", None)
            openrouter_api_key = decoded_token.get("openrouter_api_key", None)

            token_headers_str = decoded_token.get("headers", None)
            if token_headers_str:
                try:
                    token_headers = json.loads(token_headers_str)
                except json.JSONDecodeError:
                    token_headers = None
                    logger.error("Error parsing token headers")
                except Exception as e:
                    logger.error("Error parsing token headers")


            # Set API key based on endpoint, OpenAI or OpenRouter
            if chat_completion.use_openrouter == True and openrouter_api_key:
                api_key = openrouter_api_key
            elif chat_completion.use_openrouter == False and openai_api_key:
                api_key = openai_api_key
            elif chat_completion.use_openrouter == True and not openrouter_api_key:
                raise HTTPException(status_code=401, detail="Invalid or missing OpenRouter api key")
            
        
        else:
            if chat_completion.openrouter_api_key and chat_completion.use_openrouter:
                api_key = chat_completion.openrouter_api_key
            elif chat_completion.openai_api_key and not chat_completion.use_openrouter:
                api_key = chat_completion.openai_api_key

        headers = {'Content-Type': 'application/json'}

        # If an API key is set, add it to the headers
        if api_key:
            headers["Authorization"] = f'Bearer {api_key}'

        # if user included headers in token, merge them with these ones
        if token_headers:
            headers = headers | token_headers
        logger.debug(f"Headers: {headers}")

        function_call = None
        if chat_completion.function_call:
            # If function_call is set, parse it or use it as-is
            try:
                function_call = json.loads(chat_completion.function_call)
            except json.JSONDecodeError:
                function_call = chat_completion.function_call
            except Exception as e:
                logger.error("Error parsing function call")
                logger.error(e, exc_info=True)

        data_dict = {
            "messages": await prepare_openai_messages(chat_completion),
            "model": chat_completion.model,
            "stream": True
        }

        if chat_completion.max_output_tokens:
            data_dict["max_tokens"] = chat_completion.max_output_tokens

        if chat_completion.functions and chat_completion.function_call:
            data_dict["functions"] = chat_completion.functions
            data_dict["function_call"] = function_call
        
        data = json.dumps(data_dict)

        logger.debug("Data: ")
        logger.debug(data)

        return url, headers, data
    
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error("Error preparing OpenAI request:", exc_info=True)
        raise e

async def handle_exception(e):
    logger.error("Error processing chat completion:", exc_info=True)
    # This will include a traceback and additional info about the exception
    error_details = {
        "type": str(type(e)),
        "message": str(e),
        "trace": traceback.format_exc()
    }
    logger.error(error_details)

    if hasattr(e, "detail"):
        message = str(e.detail)
    elif hasattr(e, "message"):
        message = str(e.message)
    else:
        message = "There was a problem with your request.  Please check your request and try again."

    return JSONResponse(
        status_code=e.status_code if hasattr(e, "status_code") else 400,
        content={
            "status_code": 400,
            "error": message
    })


# Decode request body for function call
async def parse_function_call(chat_completion):
    try:
        # Try to parse the string as JSON
        function_call = json.loads(chat_completion.function_call)
        return function_call
    except json.JSONDecodeError:
        # If it fails, just return the original string
        return chat_completion.function_call
    except Exception as e:
        logger.error("Error parsing function call")
        logger.error(e, exc_info=True)
        return chat_completion.function_call

# Set a maximum number of retries for the API call
max_retries = 3

# Generate and yield the response from the API call
# This is POST to openai wrapper
async def event_generator(url, headers, data):
    logger.debug("Starting event generator")
    logger.debug(data)

    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:  # Increased timeout
                async with client.stream('POST', url, headers=headers, data=data) as response:
                    logger.debug(f"Response received from API - {url}")

                    if response.status_code != 200:
                        try:
                            
                            response_body = await response.aread()
                            response_json = json.loads(response_body)
                            logger.debug(response_json)

                            response_message = response_json.get("error", {}).get("message", "Error processing Chat Completion")
                            error_type = response_json.get("error", {}).get("code", "unclassified_openai_error")

                            error_message = json.dumps({
                                "status": response.status_code,
                                "error": response_message,
                                "type": error_type
                            })

                            yield error_message 
                            break
                        except Exception as e:
                            logger.error("Error handling OpenAI response", exc_info=True)
                            error_message = json.dumps({
                                "status": e.status_code if hasattr(e, "status_code") else 500,
                                "message": e.detail if hasattr(e, "detail") else "Internal server error"
                            })
                            yield error_message
                    else:
                        # Process successful response
                        async for item in handle_stream(response):
                            logger.debug(item)
                            yield item
                            
                    break  # Successful, so break the loop

        except httpx.ReadTimeout:
            logger.warning(f"Timeout on attempt {attempt + 1} out of {max_retries}. Retrying...")

        except Exception as e:
            logger.error("Error in chat completion event_generator:", exc_info=True)
            yield json.dumps({
                "status": 500,
                "error": "Internal server error"
            })
            break  # Stop retrying if an unknown exception occurs



async def handle_stream(response):
    buffer = ""
    async for chunk in response.aiter_text():
        logger.debug(f'Chunk: {chunk}')
        buffer += chunk
        while '\n\n' in buffer:  # Check if there are complete messages in buffer
            line, buffer = buffer.split('\n\n', 1)  # Split buffer into line and remaining buffer
            if line:
                yield line + "\n\n"

async def num_tokens_from_single_message(message, model="gpt-3.5-turbo-0301"):
    ''' Returns the number of tokens used by a single message.
    This is a wrapper function for num_tokens_from_messages.
    '''
    return await num_tokens_from_messages([message], model)

async def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    ''' Returns the number of tokens used by a list of messages.
    This is used to check if the token limit has been exceeded.
    Uses the tiktoken library to encode the messages.

    Adapted from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    '''

    logger.debug(f"Calculating number of tokens for {len(messages)} message")

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        logger.warn("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")

    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        logger.debug("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return await num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        logger.debug("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return await num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        # NOTE - this is a catch-all for new models. Update as needed.
        return await num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
        # raise NotImplementedError(
        #     f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        # )

    num_tokens = 0
    for message in messages:
        logger.debug("Counting tokens in message:")
        logger.debug(message)

        num_tokens += tokens_per_message

        if 'function_call' in message:
            logger.debug("Skipping function call message")
            break

        for key, value in message.items():
            if type(value) == dict:  # for function calls
                value = json.dumps(value)
            if key == "name":
                num_tokens += tokens_per_name
            if key == "function_call":
                break
            # Every assistant message is primed with <|start|>assistant<|message|>
            if message["role"] == "assistant" and key == "content":
                num_tokens += 3
            # For messages with image url
            if isinstance(message['content'], list):  # Check if content is a list
                # Initialize value as None or a default text
                value = None
                # Iterate through each item in the content list
                for item in message['content']:
                    # Check if the item is of type 'text'
                    if item.get('type') == 'text':
                        # Extract the text from the item
                        value = item.get('text')
                        break  # Break the loop after finding the first text item
            num_tokens += len(encoding.encode(value))
    logger.debug(f"Number of tokens calculated as: {num_tokens}")
    return num_tokens

async def set_max_tokens_by_model(model: str) -> int:

    # assume gpt-3.5 baseline
    max_tokens = 4096  

    # Set the max tokens based on the model
    if model == "gpt-3.5-turbo":
        max_tokens = 16384
    if model == "gpt-3.5-turbo-16k":
        max_tokens = 16384
    elif model == "gpt-4":
        max_tokens = 8192
    elif model == "gpt-4-32k":
        max_tokens = 32768
    elif model.startswith("gpt-4-turbo"):
        max_tokens = 128000
    elif model.startswith("gpt-4o"):
        max_tokens = 128000
    elif model in [
        "gpt-4-vision-preview", 
        "gpt-4-0125-preview", 
        "gpt-4-1106-preview"
    ]:
        max_tokens = 128,000
    
    return max_tokens


async def trim_single_message(message: dict, max_tokens: int, model):
    ''' Helper method to trim a single message '''

    logger.debug("Trimming single message")

    # Determine the token count for the given message
    token_count = await num_tokens_from_messages([message], model)
    excess_tokens = token_count - max_tokens

    if excess_tokens <= 0:
        return message  # no trimming needed

    # Calculate the ratio of excess tokens to the total token count for the message content
    trim_ratio = excess_tokens / token_count

    # HACK - for cases where the message content is a list, use the first text item
    if type(message["content"]) == list:
        logger.debug("Message content is a list")
        for msg in message["content"]:
            if msg.get("type", None) == "text":
                message = {"role": "user", "content": msg["text"]}
                break
        

    # Initial trim using the ratio
    trim_length = int(len(message["content"]) * trim_ratio)
    trimmed_content = message["content"][:-trim_length]

    # Binary search based refining
    left, right = 0, trim_length
    while left <= right:
        mid = (left + right) // 2
        trimmed_content = message["content"][:-mid]

        trimmed_message = message.copy()
        trimmed_message["content"] = trimmed_content

        current_tokens = await num_tokens_from_messages([trimmed_message], model)
        if current_tokens > max_tokens:
            left = mid + 1
        else:
            right = mid - 1

    # Final adjustment: Use the right pointer which should give the closest trim to the desired token count.
    trimmed_message["content"] = message["content"][:-right]

    logger.debug("Trimmed message: " + str(trimmed_message))
    return trimmed_message

async def num_tokens_from_each_message(messages, model="gpt-3.5-turbo-0301"):
    ''' Returns a list of token counts corresponding to each message in the batch.
    This utilizes the num_tokens_from_single_message function for efficiency.
    '''
    individual_token_counts = []
    for message in messages:
        individual_token_counts.append(await num_tokens_from_single_message(message, model))
    return individual_token_counts

async def auto_truncate_messages(messages, model="gpt-3.5-turbo-0301", max_input_tokens=None) -> list[dict]:

    if not messages:
        # Handle the empty case appropriately
        return []

    try:
        logger.debug(f"Auto truncating {len(messages)} messages")
        
        # Remove messages with empty content
        #messages = [m for m in messages if m["content"].strip() != "" or m["function_call"] is not None]
        messages = [
            m for m in messages if m.get("content", "") or m.get("function_call", "")
        ]
        individual_token_counts = [await num_tokens_from_single_message(msg, model) for msg in messages]

        total_tokens = sum(individual_token_counts)
        
        logger.debug("Total Token count: " + str(total_tokens))
        
        model_max_tokens = await set_max_tokens_by_model(model)

        # Most models allow 4096 response tokens
        if model_max_tokens > 4096:
            default_usable_model_max_tokens = model_max_tokens - 4096
        else:
            default_usable_model_max_tokens = model_max_tokens // 2

        logger.debug(f"Model max tokens: {model_max_tokens}. Default usable model max tokens: {default_usable_model_max_tokens}.")

        if max_input_tokens is not None:
            max_tokens = min(default_usable_model_max_tokens, max_input_tokens)
        else:
            max_tokens = default_usable_model_max_tokens

        logger.debug("Max tokens: " + str(max_tokens))

        # Identify system and last user message
        system_msg = next((msg for msg in messages if msg['role'] == 'system'), None)
        last_user_msg = messages[-1] if messages[-1]['role'] == 'user' else None

        # Create empty list for truncated messages
        truncated_messages = []

        # Handle trimming of the system message if necessary
        if system_msg:
            idx = messages.index(system_msg)
            truncated_messages.append(system_msg)
            total_tokens -= individual_token_counts[idx]
            messages.remove(system_msg)

        # New logic to remove oldest messages
        # Avoid directly removing system and last user messages
        while total_tokens > max_tokens:
            msg_to_remove = next((msg for msg in messages if msg != last_user_msg and msg != system_msg), None)
            if msg_to_remove:
                idx_to_remove = messages.index(msg_to_remove)
                removed_tokens = individual_token_counts.pop(idx_to_remove)
                messages.pop(idx_to_remove)
                total_tokens -= removed_tokens
            else:
                break

        # Handle trimming of the last_user_msg if necessary
        if total_tokens > max_tokens and last_user_msg:
            excess_tokens = total_tokens - max_tokens
            last_user_msg_token_count = await num_tokens_from_single_message(last_user_msg, model)
            desired_token_count_for_last_msg = last_user_msg_token_count - excess_tokens
            trimmed_last_user_msg = await trim_single_message(last_user_msg, desired_token_count_for_last_msg, model)
            
            # Remove the original (untrimmed) last_user_msg from the messages list
            if last_user_msg in messages:
                messages.remove(last_user_msg)
            
            # Now append the trimmed version to the main messages list
            messages.append(trimmed_last_user_msg)
        
        # Add remaining messages back to truncated_messages
        truncated_messages.extend(messages)

        logger.debug(f"Token limit not exceeded- returning {len(truncated_messages)} messages")

        logger.debug(f"Returning {len(truncated_messages)} truncated messages")
        return truncated_messages

    except Exception as e:
        logger.error("Error auto truncating message history")
        logger.error(e, exc_info=True)
        return messages


async def call_api(function_call: FunctionCall) -> str:
    ''' Use the requests library to call the API requested by the function call.

    FunctionCall model provides url, method, headers, and body.
    '''
    
    async with httpx.AsyncClient() as client:

        #method, url, headers, body = await prepare_request_details(function_call)

        try:
            # Extract args
            args = json.loads(function_call.function_call_body).get("arguments", {})

            headers = json.loads(function_call.headers) if function_call.headers else None
            logger.debug("Request headers:")
            logger.debug(headers)
            response: httpx.Response = await client.request(
                method=function_call.method,
                url=function_call.api_url,
                headers=headers,
                data=args
            )
            logger.debug(response)
        except Exception as e:
            logger.error("Error calling API:", exc_info=True)
            raise HTTPException(status_code=500, detail="Error calling API")

        return await handle_api_response(response, function_call)

        
async def prepare_request_details(function_call: FunctionCall):
    ''' Prepare the request details based on the provided function call '''

    method = function_call.method

    # Ensure URL starts with http or https protocol
    url = function_call.api_url
    if not url.startswith("http"):
        url = "https://" + url

    # Convert headers to dict if provided
    headers = json.loads(function_call.headers) if function_call.headers else None

    # If "arguments" are present in function_call and method is POST, construct the body for the API call
    if function_call.function_call_body and method == "POST":
        body = await parse_function_call(function_call)

    #body = await construct_body(function_call) if function_call.function_call_body and method == "POST" else None

    return method, url, headers, body


async def handle_api_response(response: httpx.Response, function_call: FunctionCall) -> str:
    ''' Handle the API response, either logging the error or returning the response body '''

    if response.status_code >= 400:
        logger.error(f"Error on from function call API: {response}")
        error_response = {
            "error": f"Error {response.status_code}: {response.reason_phrase}",
            "details": response.text
        }
        return json.dumps(error_response)

    if response.headers.get("content-type") == "application/json":
        return json.dumps(response.json())
    
    return response.text
