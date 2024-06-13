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

async def decrypt_jwt(encrypted_jwt: str):
    """
    Decrypt the encrypted JWT and return the decrypted JWT.
    """
    try:
        logging.debug(f"Decrypting JWT: {encrypted_jwt}")

        # Decode the key from base64
        key_string = os.getenv("JWT_ENCRYPTION_KEY", None)
        encryption_key = key_string.encode('utf-8')

        # Create the Fernet object to decrypt
        f = Fernet(encryption_key)

        # decrypt payload
        decrypted_jwt = f.decrypt(encrypted_jwt.encode())
        logging.debug(f'Decrypted JWT: {decrypted_jwt.decode()}')

        # Remove the marker
        decrypted_jwt = decrypted_jwt.decode()
        parsed_jwt = decrypted_jwt[11:]

        logging.debug(f"Decrypted JWT: {parsed_jwt}")

        return decrypted_jwt
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error decrypting JWT")

async def generate_token(request: IssueTokenRequest):
    try:
        expire = datetime.utcnow() + timedelta(minutes=request.ttl)
        data = {
            "api_key": request.openai_api_key,  # note the legacy
            "openrouter_api_key": request.openrouter_api_key,
            "exp": expire,
            "password": request.password,
            "headers": request.headers
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
    logger.debug("Verifying token")
    logger.debug(f'token: {token}')
    logger.debug(f'app_url: {app_url}')

    try:

        # Step 1: verify signature
        decoded_token = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=["HS256"],
            options={"verify_aud": False}
        )

        # Step 2: check if aud claim is present
        aud_claim = decoded_token.get('aud', None)

        if aud_claim:
            # Step 3: verify aud claim
            if normalize_url(aud_claim) != normalize_url(app_url):
                raise jwt.InvalidAudienceError("Invalid audience")

        # verify not blacklisted
        logger.debug("Trying to verify token is not blacklisted")
        if await redis_manager.get_value(token) == "BLACKLISTED":
            raise jwt.InvalidTokenError("Token is no longer valid")

        return decoded_token
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Signature has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
    # except jwt.InvalidAudienceError:
    #     raise HTTPException(status_code=401, detail="Invalid audience")
