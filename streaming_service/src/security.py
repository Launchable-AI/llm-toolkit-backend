from .log_config import logging
logger = logging.getLogger(__name__)

import os
from datetime import datetime, timedelta
import jwt
from urllib.parse import urlparse
import base64

from .models import IssueTokenRequest
from .redis_manager import redis_manager

from fastapi import HTTPException

from cryptography.fernet import Fernet

from .config import settings

import requests

def normalize_url(url):
    """
    Normalize the URL by removing the scheme (http/https) and 'www' if present.
    """
    # Parse the URL to remove the scheme
    parsed_url = urlparse(url)
    hostname = parsed_url.netloc or parsed_url.path  # netloc for complete URL, path for without scheme

    # Remove 'www.' if it exists
    if hostname.startswith('www.'):
        hostname = hostname[4:]

    return hostname

async def encrypt_jwt(jwt: str):
    """
    Encrypt the JWT and return the encrypted JWT.
    """
    try:
        logging.debug(f"Encrypting JWT: {jwt}")

        # Decode the key from base64
        key_string = os.getenv("JWT_ENCRYPTION_KEY", None)
        encryption_key = key_string.encode('utf-8')

        # Create the Fernet object to encrypt
        f = Fernet(encryption_key)

        # encrypt payload
        encrypted_jwt = f.encrypt(jwt.encode())
        logging.debug(f"Encrypted JWT: {encrypted_jwt.decode()}")

        # Concatenate a marker to indicate its encrypted
        security_marker = "azx09ap29t"
        return_payload = security_marker + encrypted_jwt.decode()
        logging.debug(f"Return payload: {return_payload}")

        return return_payload

    except Exception as e:
        raise HTTPException(status_code=500, detail="Error encrypting JWT")

async def decode_jwt(token: str):
    """
    Decode the JWT and return the decoded JWT.
    """
    try:
        logging.debug(f"Decoding JWT: {token}")

        decoded_jwt = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=["HS256"],
            options={"verify_aud": False}
        )

        logger.debug(f"Decoded JWT: {decoded_jwt}")

        return decoded_jwt

    except Exception as e:
        raise HTTPException(status_code=500, detail="Error decrypting JWT")

async def remove_jwt_marker(jwt_with_marker: str):
    ''' Remove the marker from the encrypted JWT
    '''

    # Check if the marker is present
    security_marker = "azx09ap29t"
    marker_length = len(security_marker)
    # if not jwt_with_marker.startswith(security_marker):
    #     raise HTTPException(status_code=400, detail="Invalid JWT")

    # Remove the marker
    encrypted_jwt_no_marker = jwt_with_marker[marker_length:]

    return encrypted_jwt_no_marker

async def decrypt_jwt(raw_security_token: str):
    """
    Wrapper to decrypt and decode the encrypted and prefixed JWT.

    We assume the encrypted JWT is prefixed with a marker to indicate it is encrypted.

    1. Remove the marker
    2. Decrypt the JWT
    3. Decode the JWT
    4. Return the decoded JWT

    """
    try:
        logging.debug(f"Decrypting JWT: {raw_security_token}")

        # 1. Remove the marker
        encrypted_jwt = await remove_jwt_marker(raw_security_token)

        # Decode the key from base64
        key_string = os.getenv("JWT_ENCRYPTION_KEY", None)
        encryption_key = key_string.encode('utf-8')

        # Create the Fernet object to decrypt
        f = Fernet(encryption_key)

        # 2. decrypt payload
        decrypted_jwt_bytes = f.decrypt(encrypted_jwt.encode())
        decrypted_jwt = decrypted_jwt_bytes.decode()
        logger.debug(f'Decrypted JWT: {decrypted_jwt}')

        # 3. Decode the JWT
        decoded_jwt = await decode_jwt(decrypted_jwt)

        # # # Remove the first 11 characters which are the marker
        # # parsed_jwt = decoded_jwt[11:]

        # # logging.debug(f"Returning: {parsed_jwt}")

        return decoded_jwt

    except Exception as e:
        logger.error(f"Error decrypting JWT: {e}")
        raise HTTPException(status_code=500, detail="Error decrypting JWT")

async def generate_token(request: IssueTokenRequest):
    try:
        expire = datetime.utcnow() + timedelta(minutes=request.ttl)
        data = {
            "api_key": request.openai_api_key,  # note the legacy
            "openrouter_api_key": request.openrouter_api_key,
            "exp": expire,
            "password": request.password,
            "headers": request.headers,
            "user_id": request.user_id,
            "request_limit": request.request_limit,
            "llm_toolkit_key": request.llm_toolkit_key,
            }

        if request.app_url:
            data["aud"] = normalize_url(request.app_url)
        jwt_secret_key = settings.jwt_secret_key
        
        # sign the jwt
        encoded_jwt = jwt.encode(data, jwt_secret_key, algorithm="HS256")

        # encrypt the jwt
        encrypted_jwt = await encrypt_jwt(encoded_jwt)

        logger.debug("Generated token")
        logger.debug(encoded_jwt)
        logger.debug("From data:")
        logger.debug(data)
        logger.debug("Encrypted token:")
        logger.debug(encrypted_jwt)

        #return encoded_jwt
        return encrypted_jwt

    except Exception as e:
        raise HTTPException(status_code=500, detail="Error generating token")

async def verify_token(token: str, app_url: str = None):
    ''' Verify the token

    Raise errors if the token is invalid

    Invalid states:
    - token is expired
    - token is invalid
    - token has invalid audience
    - token is blacklisted
    - request limit exceeded

    '''

    logger.debug("Verifying token")
    logger.debug(f'token: {token}')
    logger.debug(f'app_url: {app_url}')

    try:

        aud_claim = token.get('aud', None)

        if aud_claim:
            # Step 3: verify aud claim
            if normalize_url(aud_claim) != normalize_url(app_url):
                raise jwt.InvalidAudienceError("Invalid audience")

        # verify not blacklisted
        # logger.debug("Trying to verify token is not blacklisted")
        # if await redis_manager.get_value(token) == "BLACKLISTED":
        #     raise jwt.InvalidTokenError("Token is no longer valid")

        # Step 4: verify request_limit not exceeded
        # request_limit = decoded_token.get('use_count', None)
        # if request_limit is not None:
        #     # fetch uses remaining from Bubble
        #     uses_remaining = await requests.get(
        #     if int(request_limit) <= 0:
        #         raise jwt.InvalidTokenError("Request limit exceeded")

        return True
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Signature has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
    # except jwt.InvalidAudienceError:
    #     raise HTTPException(status_code=401, detail="Invalid audience")
