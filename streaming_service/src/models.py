from typing import List, Any
from pydantic import BaseModel, Field
from fastapi import UploadFile, File

class ChatCompletion(BaseModel):
    #prompt: str
    model: str = "gpt-3.5-turbo"
    messages: List[dict]   # double check this
    security_token: str | None
    openai_api_key: str | None
    max_input_tokens: int | None # For auto-truncation - controls how much of the message history left for output
    max_output_tokens: int | None # Controls how many tokens are generated
    smart_truncate: bool = True
    temperature: float | None
    stop_sequences: str | None
    function_call: str | dict | None = None
    functions: list | None = None
    user: str | None = None
    top_p: float | None = None
    logit_bias: str | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    last_n_messages: int | None = None
    image_url_1: str | None = None
    image_url_2: str | None = None
    app_url: str | None = None
    use_openrouter: bool = False
    openrouter_api_key: str | None = None
    custom_endpoint: str | None = None
    custom_headers: dict | None = None
    response_format: dict | None = None
    hidden_context: str | None = None
    custom_body: str | dict | None = None

class TokenCountRequest(BaseModel):
    messages: List[dict]
    model: str = "gpt-3.5-turbo"

class FunctionCall(BaseModel):
    model: str = "gpt-3.5-turbo"
    openai_api_key: str | None
    api_url: str | None
    function_call_body: str | None # will be None if a GET request
    headers: str | None = None
    custom_plugin_url: str | None = None
    method: str = "POST"
    test_field: str | None = "TESTTESTTEST"

class ResponseMessage(BaseModel):
    message_text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = "gpt-3.5-turbo"

class IssueTokenRequest(BaseModel):
    openai_api_key: str | None = None
    openrouter_api_key: str | None = None
    ttl: int = 1800
    password: str | None = None
    app_url: str | None = None
    headers: str | None = None
    #caller_id: str  # is context.currentUser._id and _createdDate

class ExpireTokenRequest(BaseModel):
    token: str
    password: str | None = None

class TruncationRequest(BaseModel):
    messages: List[dict] | None
    model: str = "gpt-3.5-turbo"
    max_input_tokens: int | None = None

class TruncationStandaloneRequest(BaseModel):
    messages: str | None
    model: str = "gpt-3.5-turbo"
    max_input_tokens: int | None = None

class PostTestingDebugging(BaseModel):
    sample_text: str | None

# class TranscribeRequest:
#     file: UploadFile = File(...)
#     model: str = "whisper-1"
#     language: str = "en"

class CreateThreadAndRunRequest(BaseModel):
    assistant_id: str | None = None
    thread: dict | None = None
    run_metadata: dict | None = None
    tools: list | None = None
    model: str | None = None
    instructions: str | None = None
    user_openai_api_key: str | None = None
    security_token: str | None = None
    app_url: str | None = None
    stream: bool | None = True

class CreateRunRequest(BaseModel):
    assistant_id: str | None = None
    run_metadata: dict | None = None
    tools: list | None = None
    model: str | None = None
    instructions: str | None = None
    user_openai_api_key: str | None = None
    security_token: str | None = None
    app_url: str | None = None
    stream: bool | None = True
