from src.log_config import get_logger
logger = get_logger(__name__)

import os
import asyncio
import jwt

import src.router as router
from src.config import front_end_limiter
from src.security import decrypt_jwt, decode_jwt, generate_token

import httpx

from src.models import IssueTokenRequest

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from starlette.requests import Request
from pydantic import ValidationError


app = FastAPI()

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.debug("Error validating request:")
    logger.debug(exc)

    return JSONResponse(
        status_code=422,
        content={
            "message": f"Your workflow has missing or invalid field(s).  Please correct this and try again.",
            "status_code": 422 # including in body b/c Bubble doesn't get the response status code 
        }
    )

app.state.limiter = front_end_limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow only this origin
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Allow only GET and POST methods
    allow_headers=["Authorization", "Content-Type"],
)

BUBBLE_API_URL = os.getenv("BUBBLE_API_URL", None)
BUBBLE_API_TOKEN = os.getenv("BUBBLE_API_TOKEN", None)

async def receive_body(request: Request):
    ''' Helper to cache the request body for logging
    '''
    return { "type": "http.request", "body": await request.body() }

# Replace with your Bubble Data API URL and API token
async def log_request_to_bubble(request: Request, jwt=None):
    """
    """

    # Heart attack...
    logger.debug("More heartattacks")

    logger.debug("Request passed to log_request_to_bubble")
    logger.debug(request)

    try:
        # Parse the JSON body
        request_body = await request.json()
        logger.debug(f"Request body passed to log func: {request_body}")
        logger.debug(f"jtw passed to log func: {jwt}")

        # Get the user_id, aud, and JWT from the request
        user_id = request_body.get("user_id", "")
        app_url = request_body.get("app_url", "")
        security_token_text = jwt if jwt else None

        # Get the request details
        route = getattr(request.url, 'path', "unknown") # endpoint being called

        # Domain name and IP address
        host = request.headers.get('host', 'unknown')
        logger.debug(request.headers)
        x_forwarded_for = request.headers.get('x-forwarded-for', 'unknown')
        ip_address = getattr(request.client, 'host', 'unknown')

        # Prepare the data to send to Bubble
        data = {
            "app_url_text": app_url,
            "route_text": route,
            "host_text": host,
            "x_forwarded_for_text": x_forwarded_for,
            "ip_address_text": ip_address,
            "security_token_text": security_token_text,
            "user_id_text": user_id,
        }

        logger.debug(f"Logging request to Bubble: {data}")

        # Send the data to Bubble's Data API
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    BUBBLE_API_URL,
                    json=data,
                    headers={"Authorization": f"Bearer {BUBBLE_API_TOKEN}"},
                )
                logger.debug(f"Logged request to Bubble: {response.status_code} - {response.text}")
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTPStatus Error: Failed to log request to Bubble: {e}")
            except Exception as e:
                logger.error("Unclassified error")
                logger.error(f"An unexpected error occurred while logging to Bubble: {e}")

    except Exception as e:
        logger.error(f"An unexpected error occurred while logging to Bubble: {e}")


# @app.middleware("http")
# async def log_middleware(request: Request, call_next):

#     logger.debug ("Test string")

#     request_body_bytes = await request.body()
#     request_body_str = request_body_bytes.decode("utf-8")
#     logger.debug(f"Request body - pre logging: {request_body_str}")

#     # Create a new Request object with the cached body
#     new_request = Request(request.scope, receive=lambda: receive_body(request))

#     response = await call_next(new_request)

#     # Log the request asynchronously
#     try:
#         # if request is to /issue_token, re-create the token and log it
#         logger.debug(f"Request URL: {request.url.path}")
#         if request.url.path == "/issue_token":
#             try:
#                 request_json = await request.json()
#                 issue_token_request = IssueTokenRequest.parse_obj(request_json) 
#                 jwt = await generate_token(issue_token_request)
#                 logger.debug(f"Generated token: {jwt}")

#                 # Log the request
#                 await asyncio.create_task(log_request_to_bubble(new_request, jwt))

#             except ValidationError as e:
#                 logger.error(f"Validation error: {e}")
#                 raise HTTPException(status_code=400, detail="Invalid request body")
#             except Exception as e:
#                 logger.error(f"Error generating token: {e}")
#                 raise HTTPException(status_code=500, detail="Internal server error")

#         else:
#             asyncio.create_task(log_request_to_bubble(new_request))
#     except Exception as e:
#         logger.error(f"Logging to Bubble failed: {e}", exc_info=True)

#     return response

# @app.middleware("http")
# async def log_middleware(request: Request, call_next):

#     request_body = await request.body()
#     logger.debug(f"Request body - pre logging: {request_body}")

#     # Create a new Request object with the cached body
#     new_request = Request(request.scope, receive=lambda: receive_body(request))

#     response = await call_next(new_request)

#     # Log the request asynchronously
#     try:
#         # if request is to /issue_token, re-create the token and log it
#         logger.debug(f"Request URL: {request.url.path}")
#         if request.url.path == "/issue_token":
#             request_json = await request.json()
#             issue_token_request = IssueTokenRequest.parse_obj(request_body) 
#             jwt = await generate_token(issue_token_request)
#             logger.debug(f"Generated token: {jwt}")

#             # Log the request
#             await asyncio.create_task(log_request_to_bubble(new_request, jwt=None))
#         asyncio.create_task(log_request_to_bubble(new_request))
#     except Exception as e:
#         logger.error(f"Logging to Bubble failed: {e}", exc_info=True)

#     return response

app.include_router(router.router)