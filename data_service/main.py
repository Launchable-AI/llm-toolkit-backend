import src.router as router

from fastapi import FastAPI

app = FastAPI()

app.include_router(router.router)