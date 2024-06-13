from .log_config import logging
logger = logging.getLogger(__name__)

import os
import json
import jwt
from typing import Optional

from .models import ChatCompletion, IssueTokenRequest, ExpireTokenRequest, \
    TokenCountRequest, FunctionCall, TruncationRequest, PostTestingDebugging, \
    TruncationStandaloneRequest, CreateThreadAndRunRequest, CreateRunRequest
from .security import generate_token, decrypt_jwt, verify_token
from .helpers import prepare_openai_request, event_generator,\
    handle_exception, call_api, num_tokens_from_messages, auto_truncate_messages
from .redis_manager import redis_manager
from .config import settings, front_end_limiter, back_end_limiter

from openai import OpenAI, AsyncOpenAI

from fastapi import HTTPException, UploadFile, File, Form

import requests 
from fastapi import Request, APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from slowapi.errors import RateLimitExceeded
from starlette.requests import Request

router = APIRouter()

@router.on_event("startup")
async def startup():
    await redis_manager.startup()

@router.on_event("shutdown")
async def shutdown():
    await redis_manager.shutdown()

@router.get("/streaming/ping")
@router.get("/ping")
async def ping():
    return {
        "message": "pong",
        "status_code": 200
    }

@router.get("/streaming/testing_debugging")
@router.get("/testing_debugging")
async def get_testing_debugging(
    sample_text: str = "sample text not set"
    ):
    ''' A sample endpoint for testing/debugging

    Args:
        - sample_text (str): Just a string to ensure args are being passed correctly

    Returns:
        - JSONResponse: The response body
    '''
    logger.info(f"Request to /get_testing_debugging")
    logger.info(f"Details: {sample_text}")

    return JSONResponse(
        status_code=200,
        content={
            "message": f"Testing debugging endpoint with sample_text: {sample_text}",
            "status_code": 200 # including in body b/c Bubble doesn't get the response status code 
        })



@router.post("/streaming/testing_debugging")
@router.post("/testing_debugging")
async def post_testing_debugging(
    post_testing_debugging: PostTestingDebugging
    ):
    try:
        logger.info(f"POST Request to /testing_debugging")
        logger.info(post_testing_debugging)

        return JSONResponse(
            status_code=200,
            content={
                "message": f"Testing POST debugging endpoint with sample_text: {post_testing_debugging.sample_text}",
                "status_code": 200 # including in body b/c Bubble doesn't get the response status code 
            })
    except Exception as e:
        return await handle_exception(e)

        

@router.post("/streaming/chat_completion")
@router.post("/chat_completion")
#@front_end_limiter.limit("30/minute")
async def chat_completion( 
        request: Request,
        chat_completion: ChatCompletion
    ):
    try:
        logger.info(f"Request to /chat_completion")
        logger.debug(chat_completion)
        logger.debug("Request data:")
        logger.debug(request)

        url, headers, data = await prepare_openai_request(chat_completion)
        logger.debug(f"URL: {url}")
        logger.debug(f"Headers: {headers}")
        logger.debug(f"Data: {data}")

        # If custom body is provided, overwrite the compiled data
        if chat_completion.custom_body:
            data = chat_completion.custom_body
            logger.debug(f"custom_body: {chat_completion.custom_body}")

        return StreamingResponse(event_generator(url, headers, data), media_type="text/event-stream")
    except RateLimitExceeded as e:
        return JSONResponse(
            status_code=429,
            content={
                "message": f"Your workflow has exceeded the rate limit.  Please try again later.",
                "status_code": 429 # including in body b/c Bubble doesn't get the response status code 
            }
        )
    except Exception as e:
        return await handle_exception(e)

@router.post("/streaming/auto_truncate")
@router.post("/auto_truncate")
#@front_end_limiter.limit("30/minute")
async def smart_truncate(
    request: Request,
    truncation_request: TruncationRequest
):
    ''' Truncate messages to model limit
    '''

    try:
        logger.debug(f"Request to /auto_truncate for {len(truncation_request.messages)} messages")

        truncated_messages = await auto_truncate_messages(
            truncation_request.messages, 
            truncation_request.model, 
            truncation_request.max_input_tokens
            )
        
        return JSONResponse(
            status_code=200,
            content={
                "truncated_messages": truncated_messages,
                "status_code": 200 # including in body b/c Bubble doesn't get the response status code 
            })

    except Exception as e:
        logger.error("Error auto-truncating messages:")
        logger.error(e, exc_info=True)
        return JSONResponse(
            status_code=400,
            content={
                "message": "Failed to auto-truncate messages.  Please try again or contact support.",
                "status_code": 400 # including in body b/c Bubble doesn't get the response status code 
            })

@router.post("/streaming/auto_truncate_standalone")
@router.post("/auto_truncate_standalone")
#@front_end_limiter.limit("30/minute")
async def smart_truncate_standalone(
    request: Request,
    truncation_request: TruncationStandaloneRequest
):
    ''' Truncate messages to model limit

    Using a new endpoint (standalone) in case we're using the earlier endpoint
    somewhere
    '''

    try:
        logger.debug(f"Request to /auto_truncate_standalone.  Data:")
        logger.debug(truncation_request)

        messages = json.loads(truncation_request.messages)

        truncated_messages = await auto_truncate_messages(
            messages, 
            truncation_request.model, 
            truncation_request.max_input_tokens
            )
        
        return JSONResponse(
            status_code=200,
            content={
                "truncated_messages": truncated_messages,
                "status_code": 200 # including in body b/c Bubble doesn't get the response status code 
            })

    except Exception as e:
        logger.error("Error auto-truncating messages:")
        logger.error(e, exc_info=True)
        return JSONResponse(
            status_code=400,
            content={
                "error_message": str(e),
                "status_code": 400 # including in body b/c Bubble doesn't get the response status code 
            })

@router.post("/streaming/get_token_counts")
@router.post("/get_token_counts")
#@front_end_limiter.limit("30/minute")
async def get_token_counts(
    request: Request,
    get_token_counts: TokenCountRequest
    ):
    try:
        logger.info(f"Request to /get_token_counts")

        input_messages = get_token_counts.messages[:-1]
        response_message = get_token_counts.messages[-1]

        # Get token counts
        input_message_tokens = await num_tokens_from_messages(input_messages, get_token_counts.model)
        response_message_tokens = await num_tokens_from_messages([response_message], get_token_counts.model)

        return JSONResponse(
            status_code=200,
            content={
                "input_message_tokens": input_message_tokens,
                "response_message_tokens": response_message_tokens,
                "status_code": 200 # including in body b/c Bubble doesn't get the response status code 
            })
    except Exception as e:
        logger.error("Error getting token counts:")
        logger.error(e, exc_info=True)
        return JSONResponse(
            status_code=400,
            content={
                "message": "Failed to get token counts.  Please try again or contact support.",
                "status_code": 400 # including in body b/c Bubble doesn't get the response status code 
            })

@router.post("/streaming/issue_token")
@router.post("/issue_token")
#@back_end_limiter.limit("10/minute")
async def issue_token(
    request: Request,
    issue_token_req: IssueTokenRequest,
    ):
    ''' Issue a token that can be used to authenticate front-end calls
    without exposing the OpenAI API key.  
    
    Previously the token was stored in Redis, but now API key is part of JWT token

    The token is returned in the response body.

    Args:
        issue_token (IssueTokenRequest): The request body
            - openai_api_key (str): The OpenAI API key
            - password (str): The password to use to delete the token
            - ttl (int): The time-to-live for the token in seconds
    Returns:    
        JSONResponse: The response body

    '''
    logger.info(f"Request to /issue_token")
    logger.info(f"Request details:")
    logger.info(request)
    logger.debug(f"Details: {issue_token_req}")

    try:
        # create a JWT token
        jwt = await generate_token(issue_token_req)

        return JSONResponse(
            status_code=201,
            content={
                "message": "Token issued successfully",
                "token": jwt,
                "status_code": 201 # including in body b/c Bubble doesn't get the response status code 
            })
    except Exception as e:
        logger.error("Error issuing token:")
        logger.error(e, exc_info=True)
        return JSONResponse(
            status_code=400,
            content={
                "message": "Failed to issue token.  Please try again or contact support.",
                "status_code": 400 # including in body b/c Bubble doesn't get the response status code 
            })

@router.post("/streaming/expire_token")
@router.post("/expire_token")
async def expire_token(
    request: Request,
    expire_request: ExpireTokenRequest
    ):
    logger.info(f"Request to /expire_token")
    try:
        # The TTL here depends on how you manage it. 
        # For simplicity, we'll set it to a few days (e.g., 30 days = 2592000 seconds).
        #ttl = 2592000
        decoded_jwt = jwt.decode(
            expire_request.token,
            settings.jwt_secret_key,
            algorithms=["HS256"]
        )
        if decoded_jwt["password"] != expire_request.password:
            raise HTTPException(status_code=401, detail="Invalid password")
        # try to fetch token, to make sure it doesn't already exist in list
        token = await redis_manager.get_value(expire_request.token)
        if token:
            return JSONResponse(
                status_code=200,
                content={
                    "message": "Token already blacklisted",
                    "status_code": 200 # including in body b/c Bubble doesn't get the response status code 
                })
        else:
            await redis_manager.set_value(expire_request.token, "BLACKLISTED")
            return JSONResponse(
                status_code=200,
                content={
                    "message": "Token blacklisted successfully",
                    "status_code": 200 # including in body b/c Bubble doesn't get the response status code 
                })
    except Exception as e:
        logger.error("Error deleting token:", exc_info=True)
        logger.error(e)
        return JSONResponse(
            status_code=500,
            content={
                "message": "There was an error deleting the token.  Please try again or contact support.",
                "status_code": 404 # including in body b/c Bubble doesn't get the response status code 
        })

# TODO - improve respose message structure - just dumping it back atm
@router.post("/streaming/execute_function_call")
@router.post("/execute_function_call")
#@back_end_limiter.limit("10/minute")
async def execute_function_call(
    function_call_request: FunctionCall,
    request: Request
    ):

    try:
        logger.info(f"Request to /execute_function_call")
        logger.debug(function_call_request)

        # Extract function name from request
        if type(function_call_request.function_call_body) is str:
            function_call_body = json.loads(function_call_request.function_call_body)
            function_name = function_call_body.get("name", "undefined")
            logger.debug(f"Function Call Name: {function_name}")

            # Extract arguments from body
            function_call_arguments = json.loads(function_call_body.get("arguments", {}))
            logger.debug("Function Call arguments")
            logging.debug(function_call_arguments)

            # Downcase keys
            args = {k.lower(): v for k, v in function_call_arguments.items()}

            # if one of the params is URL, then we need to extract the URL from the body
            if args.get("url", None):
                function_call_request.api_url = args["url"]
        else:
            function_name = "undefined"

        api_call_response = await call_api(function_call_request)
        logger.debug(api_call_response)

        # Package the response into a "function" role message
        function_message = {
            "role": "function", 
            "name": function_name, 
            "content": api_call_response
        }

        return JSONResponse(
            status_code=200,
            content={
                "message": function_message,
                "status_code": 200 # including in body b/c Bubble doesn't get the response status code 
            })

    except RateLimitExceeded as e:
        raise HTTPException(
            status_code=429,
            detail={
                "message": f"Your workflow has exceeded the rate limit. Please try again later.",
                "status_code": 429
            }
        )
    except Exception as e:
        return await handle_exception(e)


@router.post("/streaming/transcribe")
@router.post("/transcribe")
#@back_end_limiter.limit("10/minute")
async def transcribe(
    request: Request,
    #transcribe_req: TranscribeRequest,
    security_token: Optional[str] = Form(None),
    api_key: Optional[str] = Form(None),
    file: UploadFile = File(...),
    model: str = Form("whisper-1"), 
    language: Optional[str] = Form("en"),
    app_url: Optional[str] = Form(None)
    ):
    try:
        logger.info(f"Request to /transcribe")

        # Ensure that the file is an audio file
        if not file.content_type.startswith("audio/"):
            raise HTTPException(status_code=400, detail="File is not an audio file.")

        # Read the content of the audio file
        content = await file.read()

        if security_token:
            logging.debug("Token detected")
            # Verify the token
            try:
                decrypted_jwt = await decrypt_jwt(security_token[10:])
                decoded_token = await verify_token(decrypted_jwt, app_url)
                openai_api_key = decoded_token.get("api_key", None)
            except Exception as e:
                raise HTTPException(status_code=401, detail="Invalid token")
        else:
            if not api_key:
                raise HTTPException(status_code=401, detail="No API key provided")
            # for case when User-provided API key is present
            openai_api_key = api_key


        client = OpenAI(api_key=openai_api_key)

        transcript = client.audio.transcriptions.create(
            model=model,
            file=(file.filename, content, file.content_type),
            language=language
        )

        #logging.debug(transcript)
        
        return transcript

    except RateLimitExceeded as e:
        return JSONResponse(
            status_code=429,
            content={
                "message": f"Your workflow has exceeded the rate limit.  Please try again later.",
                "status_code": 429 # including in body b/c Bubble doesn't get the response status code 
            }
        )
    except Exception as e:
        logger.error("Encountered error transcribing audio:")
        logger.error(e, exc_info=True) 
        logger.error(f"Security Token: {security_token}")
        logger.error(f"API Key: {api_key}")
        logger.error(f"Model: {model}")
        logger.error(f"Language: {language}")
        return JSONResponse(
            status_code=400,
            content={
                "message": f"Failed to transcribe audio.  Reason: {e}",
                "status_code": 400 # including in body b/c Bubble doesn't get the response status code 
            }
        )
async def event_stream_generator(stream):
    """
    Asynchronous generator function to yield events from the OpenAI stream.
    """
    async for event in stream:
        # THIS WORKS!
        # js = event.json()
        # js_dict = json.loads(js)
        # _json = js_dict.get("data", None)
        # _json2 = json.dumps(_json)
        #resp = f'event: {event.event}\ndata: {_json2}\n\n'

        event_data = json.loads(event.json())
    
        # Extract the 'data' key and directly use it since it's already the right structure
        data_content = event_data.get("data", {})
        
        # Prepare the SSE formatted message
        formatted_message = f"event: {event.event}\ndata: {json.dumps(data_content, ensure_ascii=False)}\n\n"

        yield formatted_message

@router.post("/streaming/threads/runs")
@router.post("/threads/runs")
async def create_thread_and_run(
    request: Request,
    create_thread_and_run_request: CreateThreadAndRunRequest
    ):
    logger.info(f"Request to /create_thread_and_run")
    logger.debug(create_thread_and_run_request)

    if create_thread_and_run_request.security_token:
        logging.debug("Token detected")
        # Verify the token
        try:
            decrypted_jwt = await decrypt_jwt(create_thread_and_run_request.security_token[10:])
            decoded_token = await verify_token(decrypted_jwt, create_thread_and_run_request.app_url)
            logger.debug(decoded_token)
            openai_api_key = decoded_token.get("api_key", None)
        except Exception as e:
            logger.debug("Error verifying token")
            logger.debug(e)
            raise HTTPException(status_code=401, detail="Invalid token")
    else:
        if not create_thread_and_run_request.user_openai_api_key:
            raise HTTPException(status_code=401, detail="No API key provided")
        # for case when User-provided API key is present
        openai_api_key = create_thread_and_run_request.user_openai_api_key

    client = AsyncOpenAI(api_key=openai_api_key)

    stream = await client.beta.threads.create_and_run(
        thread=create_thread_and_run_request.thread,
        assistant_id=create_thread_and_run_request.assistant_id,
        tools=create_thread_and_run_request.tools,
        instructions=create_thread_and_run_request.instructions,
        model=create_thread_and_run_request.model,
        metadata=create_thread_and_run_request.run_metadata,
        stream=True
    )

    # Pass the asynchronous generator to StreamingResponse
    return StreamingResponse(event_stream_generator(stream), media_type="text/event-stream")

@router.post("/streaming/threads/{thread_id}/runs")
@router.post("/threads/{thread_id}/runs")
async def create_run(
    request: Request,
    thread_id: str,
    create_run_request: CreateRunRequest
    ):

    try:
        logger.info(f"Request to /create_run")
        logger.debug(create_run_request)

        if create_run_request.security_token:
            logging.debug("Token detected")
            # Verify the token
            try:
                decrypted_jwt = await decrypt_jwt(create_run_request.security_token[10:])
                decoded_token = await verify_token(decrypted_jwt, create_run_request.app_url)
                logger.debug(decoded_token)
                openai_api_key = decoded_token.get("api_key", None)
            except Exception as e:
                logger.debug("Error verifying token")
                logger.debug(e)
                raise HTTPException(status_code=401, detail="Invalid token")
        else:
            if not create_run_request.user_openai_api_key:
                raise HTTPException(status_code=401, detail="No API key provided")
            # for case when User-provided API key is present
            openai_api_key = create_run_request.user_openai_api_key

        client = AsyncOpenAI(api_key=openai_api_key)

        stream = await client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=create_run_request.assistant_id,
            tools=create_run_request.tools,
            instructions=create_run_request.instructions,
            model=create_run_request.model,
            metadata=create_run_request.run_metadata,
            stream=True
        )

        # Pass the asynchronous generator to StreamingResponse
        return StreamingResponse(event_stream_generator(stream), media_type="text/event-stream")
    except Exception as e:
        logger.error("Error creating run:")
        logger.error(e, exc_info=True)
        return await handle_exception(e)
