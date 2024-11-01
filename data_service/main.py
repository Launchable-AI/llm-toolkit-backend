import logging
import asyncio

import src.router as router

from fastapi import FastAPI, Request


app = FastAPI()

'''
# Enable this to see the raw body of the request;
# Doesn't currently work, b/c it consumes request
@app.middleware("http")
async def log_requests(request: Request, call_next):
    body = await request.body()
    logging.info(f"Request body: {body.decode('utf-8')}")
    response = await call_next(request)
    #return response
'''

# @app.middleware("http")
# async def log_middleware(request: Request, call_next):
#     # Log the request asynchronously in the background
#     asyncio.create_task(log_request_to_bubble(request))

#     # Proceed with the request processing
#     response = await call_next(request)
#     return response

app.include_router(router.router)
