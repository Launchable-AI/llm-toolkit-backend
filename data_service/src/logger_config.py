import os
import logging

def get_logger(name: str):
    if os.getenv("DEBUG") == "True":
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    return logging.getLogger(name)