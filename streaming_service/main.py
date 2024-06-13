from src.log_config import get_logger
logger = get_logger(__name__)

import src.router as router
from src.config import front_end_limiter

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from starlette.requests import Request


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

app.include_router(router.router)