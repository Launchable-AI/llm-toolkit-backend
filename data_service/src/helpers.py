from .logger_config import get_logger
logger = get_logger(__name__)

import os
import json
from typing import List
import requests
import aiohttp
from yarl import URL  # from aiohttp
from aiohttp import ClientTimeout, ClientSession
import re
from pathlib import Path
import uuid
import asyncio
import math

from .models import WebSearchRequest, FileParsingRequest, VectorSimilarity

from fastapi import File
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse
import openai  # for embeddings
#from unstructured.partition.auto import partition  # for document text
#from unstructured.chunking.title import chunk_by_title # for text chunking
from bs4 import BeautifulSoup  # for web scraping


async def manually_chunk_by_title(elements, min_chunk_size=256, max_chunk_size=2000):
    ''' Manually chunk a list of elements by title

    Should be added to unstructured API soon:

    https://github.com/Unstructured-IO/unstructured/issues/1185

    Args:
        elements (list[dict]): List of elements
        max_chunk_size (int): Maximum chunk size in characters

    Returns:
        chunks (list[dict]): List of chunks
            text: Text of chunk
            page: Page number of chunk if available
    '''

    chunks = []
    
    for e in elements:
        logger.debug("Element:")
        logger.debug(e)
        # Helper function to get the initial chunk structure
        def create_chunk(element):
            metadata = element.get('metadata')
            page_num = metadata.get('page_number', None)

            return {
                "text": f'{element.get("text")} \n',
                "page_number": page_num
            }
        if not chunks:
            chunks.append(create_chunk(e))
        else:
            current_chunk_text = chunks[-1]['text']
            
            if len(current_chunk_text) < min_chunk_size:
                chunks[-1]['text'] += f'{e.get("text", "")}\n'
            elif e.get('type') == 'Title' or len(current_chunk_text) > max_chunk_size:
                chunks.append(create_chunk(e))
            else:
                chunks[-1]['text'] += e.get('text', "")
        
    return chunks


async def partition_with_unstructured_api(filepath: Path):
    ''' Extract text from files using th unstructured API


    Args:
        filepath (Path): Path to file to extract text from
    '''

    try:
        # URL points to local docker container running API
        url = 'http://unstructured-api:8000/general/v0/general'  

        unstructured_api_key = os.getenv('UNSTRUCTURED_API_KEY', None)

        headers = {
            'accept': 'application/json',
            'unstructured-api-key': unstructured_api_key
        }

        data = {
            "strategy": "auto",
            "include_pagebreaks": "true",
        }

        file_data = {'files': open(filepath, 'rb')}

        response = requests.post(url, headers=headers, data=data, files=file_data)

        logger.debug(f"Response from unstructured API: {response}")

        json_response = response.json()

        chunks = await manually_chunk_by_title(json_response)

        logger.debug(f"Extracted {len(chunks)} chunks from file buffer")

        file_data['files'].close()

        return chunks
    
    except Exception as e:
        logger.error(f"Error extracting text from buffer")
        logger.error(e, exc_info=True)
        raise e


async def partition(filepath: Path):
    ''' Extract text from a file

    Args:
        filepath (Path): Path to file to extract text from

    Returns:
        chunks (list[dict]): List of chunks
            text: Text of chunk
            page: Page number of chunk
    '''
    # chunks = await partition_with_unstructured_api(file_buffer=file_buffer)
    chunks = await partition_with_unstructured_api(filepath=filepath)
    return chunks

async def compose_constraints_str(key, type, value) -> str:
    ''' Compose a stringified JSON of constraints

    Args:
        key (str): Key to filter by
        type (str): Type of constraint (in, not in, etc.)
        value (str): Value to filter by

    Returns:
        constraints_str (str): Stringified JSON of constraints
    '''
    constraints = [
        { 
            "key": key,
            "constraint_type": type,
            "value": value,
        }
    ]

    constraints_str = json.dumps(constraints)
    logger.debug(f"Composed constraints string: {constraints_str}")

    return constraints_str

async def fetch_all_data_by_constraints(
        request: VectorSimilarity, 
        constraints_str: str = None, 
        total_limit: int = None) -> List[dict]:
    ''' Fetch all data from Bubble Data API by constraints

    Optional constraints string can be passed in, otherwise it will be composed

    Args:
        request (VectorSimilarity): VectorSimilarity request object
        constraints_str (str): Stringified JSON of constraints
        limit (int): Total limit of results to fetch
    '''

    results, remaining = await fetch_data_by_constraints_from_bubble(request, constraints_str)

    if remaining <= 0:
        return results

    # Calculate the number of tasks needed based on the remaining count
    # If optional total_limit set, use that instead of remaining
    # Note that the limit per request is 100 records, imposed by Bubble
    if total_limit:
        remaining = min(remaining, total_limit - len(results))
    num_tasks = math.ceil(remaining / 100) # Assuming limit is 100

    # Create tasks to fetch the remaining data concurrently
    tasks = [fetch_data_by_constraints_from_bubble(request, constraints_str, cursor=str(i * 100)) for i in range(1, num_tasks + 1)]

    # Gather all results concurrently
    all_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Flatten the results and add them to the initial results
    for sublist in all_results:
        if not isinstance(sublist, Exception):
            results.extend(sublist[0])  # sublist[0] is the data from the tuple returned by fetch_data_by_constraints_from_bubble

    return results

#async def fetch_data_by_constraints_from_bubble(request: VectorSimilarity, constraints_str: str = None) -> List[dict]:
async def fetch_data_by_constraints_from_bubble(
        request: VectorSimilarity, 
        constraints_str: str = None, 
        cursor: int = 0) -> List[dict]:
    ''' Fetch data from Bubble Data API by constraints

    Optional constraints string can be passed in, otherwise it will be composed

    Args:
        constraints_str (str): Stringified JSON of constraints
        headers (dict): Headers to send with request
        base_url (str): Base URL of Bubble Data API
        data_type (str): Data type to fetch

    Returns:
        data (list[dict]): List of dictionaries containing the results
    '''
    try:
        data_type = request.chunk_data_type.split('custom.')[1]
        full_url = f"{request.data_api_url}/{data_type}"

        # When fetching by IDs, we'll pre-chunk and generate constraints_str
        if constraints_str is None:
            constraints_str = await compose_constraints_str(request.constraint_key, request.constraint_type, request.constraint_value)

        url_with_params = URL(full_url).with_query({
            "constraints": constraints_str,
            "cursor": cursor
        })
        logger.debug(f"Fetching data from {url_with_params}")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {request.bubble_api_key}"
        }

        async with ClientSession() as session:
            async with session.get(url_with_params, headers=headers) as response:
                if response.status != 200:
                    error_message = f"Error fetching data. Status code: {response.status}. Message: {await response.text()} URL: {url_with_params}"
                    logger.error(error_message)
                    raise ValueError(error_message)
                
                # Parse the response JSON
                data = await response.json()
                logger.debug("Bubble reponse - remaining:")
                logger.debug(data["response"]["remaining"])
                return data["response"]["results"], data["response"]["remaining"]

    except Exception as e:
        logger.error(f"Failed to fetch chunk with URL {url_with_params}. Error: {str(e)}")
        return [f"Error: {str(e)}"]

def chunkify(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

async def fetch_chunks_by_id_from_bubble(request: VectorSimilarity):
    ''' This is used specifically to break up the unique IDs into chunks of 25 
    and fetch the data from Bubble, because passing a list of 1000+ IDs to the 
    Bubble API causes an error - the GET URL becomes too long

    Uses asyncio and gather to do this in parallel.  Not needed with a single
    constraint-based request.

    Args:
        request (VectorSimilarity): VectorSimilarity request object
    
    Returns:
        final_results (list[dict]): List of dictionaries containing the results
    '''

    CHUNK_SIZE = 25

    # Create chunks of unique IDs
    id_chunks = list(chunkify(request.chunk_unique_ids, CHUNK_SIZE))

    # Create the list of tasks for asyncio.gather
    tasks = []
    for id_chunk in id_chunks:
        constraints_str = await compose_constraints_str(request.constraint_key, request.constraint_type, ','.join(id_chunk))
        tasks.append(fetch_data_by_constraints_from_bubble(request, constraints_str=constraints_str))

    # Gather the results
    all_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Flatten the results and filter out any exceptions (or you can handle them differently if needed)
    final_results = [item for sublist in all_results if not isinstance(sublist, Exception) for item in sublist]

    return final_results

async def fetch_data_from_bubble(request: VectorSimilarity):
    ''' Use the Bubble data api to fetch a list of chunks and their vectors 

    Hands off to other helper methods.

    If the user is fetching by unique IDs, this will be called multiple times
    asynchronously to fetch the data in chunks of 25 IDs at a time.

    If the user is fetching by constraints, this will be called once.

    Args:
        request (VectorSimilarity): VectorSimilarity request object 

    Returns:
        final_results (list[dict]): List of dictionaries containing the results  
    '''
    
    if request.constraint_key == "_id" or request.chunk_unique_ids:
        final_results = await fetch_chunks_by_id_from_bubble(request)
    else:
        constraints_str = await compose_constraints_str(request.constraint_key, request.constraint_type, request.constraint_value)
        final_results = await fetch_all_data_by_constraints(request, constraints_str)
        #final_results = await fetch_all_data_by_constraints(request, constraints_str, total_limit=200)

    logger.debug(f"Fetched {len(final_results)} items from bubble")
    return final_results



async def send_to_bubble_data_api(request: FileParsingRequest, result_data: dict, data_api_url: str):
    ''' Send parsed file data to Bubble Data API endpoint

    Args:
        result_data (dict): Dictionary containing the parsed file data
        data_api_url (str): URL of Bubble Data API endpoint
    '''

    try:
        # Send the data to the data API

        logger.debug(f"Sending data to {data_api_url}")

        # compose params for request
        data_api_bulk_url = f"{request.data_api_url}/{request.chunk_data_type.split('custom.')[1]}/bulk"
        logger.debug(f"Composed URL is: {data_api_bulk_url}")

        records = [ {request.chunk_text_field: text} for text in result_data["texts"] ]

        # Add embeddings if they exist
        if request.chunk_embeddings_field:
            for record, embedding in zip(records, result_data["embeddings"]):
                record[request.chunk_embeddings_field] = embedding

        # Add reference field if it exists
        if request.reference_field:
            for record in records:
                record[request.reference_field] = request.reference_id 

        # Add page_number field if it exists
        if request.chunk_page_num_field:
            for record, page_number in zip(records, result_data["page_numbers"]):
                record[request.chunk_page_num_field] = page_number

        records_as_text = '\n'.join([json.dumps(record) for record in records])

        headers = {'Content-Type': 'text/plain',
                   'Authorization': f"Bearer {request.bubble_api_key}"
                   }

        response = requests.post(data_api_bulk_url, headers=headers, data=records_as_text)

        if response.status_code == 200:
            logger.info(f"Successfully sent data to {data_api_url}")

    except Exception as e:
        logger.error(f"Error sending data to {data_api_url}")
        logger.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error sending data to {data_api_url}")



async def file_parsing_wrapper(request: FileParsingRequest):
    try:
        logger.debug(f"Received request: {request}")

        # Download the file from file_url
        response = requests.get(request.file_url)

        # If response successful, parse the file
        if response.status_code == 200:
            # Create /tmp filepath 
            ext = os.path.splitext(request.file_url)[1]
            id = uuid.uuid4()
            filepath = f"/tmp/{id}{ext}"
            with open(filepath, "wb") as f:
                # Save the file to the /tmp path
                f.write(response.content)
                # Extract the chunks from the file
                # Chunks have text: and page_number: keys
                chunks = await partition(filepath)
        else:
            logger.error(f"Failed to download file from {request.file_url}")
            raise HTTPException(status_code=500, detail=f"Failed to download file from {request.file_url}")

        # If the user wants to create embeddings, do that now
        embeddings = []
        if request.create_embeddings:
            #embeddings = await create_embeddings_for_texts([chunk['text'] for chunk in chunks], request.openai_api_key)
            embeddings = await create_embeddings_for_texts([chunk['text'] for chunk in chunks], request.openai_api_key)
            logger.debug(f"Created {len(embeddings)} embeddings for {len(chunks)} chunks")
            #logger.debug(embeddings)

        result_data = {
            "texts": [chunk['text'] for chunk in chunks],
            "page_numbers": [chunk['page_number'] for chunk in chunks],
            "embeddings": embeddings,
            "message": "File parsed successfully",
            "status_code": 200
        }

        # If there is a callback URL, send the data to it
        if request.data_api_url:
            send_results = await send_to_bubble_data_api(request, result_data, request.data_api_url)
            return send_results
        elif request.wf_api_url:
            send_results = await send_to_bubble_wf_api(request, result_data, request.wf_api_url)
            return send_results
        # Otherwise, return the data
        else: 
            return result_data

    except Exception as e:
        logger.error(f"Error processing file from {request.file_url}")
        logger.error(e, exc_info=True)


async def clean_text(text: str):
    ''' Clean text by removing HTML tags, markdown, excessive newlines, and non-printable characters.
    '''
    # Remove excessive newlines
    clean_text = re.sub('\n+', '\n', text)
    # Remove tabs
    clean_text = clean_text.replace('\t', ' ')
    # Remove repeated spaces
    clean_text = re.sub(' +', ' ', clean_text)
    # Remove non-printable characters
    clean_text = ''.join(char for char in clean_text if char.isprintable())
    return clean_text

async def fetch_page(session, url):
    ''' Asynchronously fetch a page and return its contents.

    Args:
        session: aiohttp.ClientSession
        url: URL of page to fetch
    
    Returns:
        page_content (str): Contents of page
    '''
    try:
        timeout = ClientTimeout(total=5)  
        async with session.get(url, timeout=timeout) as response:
            page_content = await response.text()
            # Remove HTML tags and markdown
            soup = BeautifulSoup(page_content, 'html.parser')
            text = soup.get_text()
            cleaned_text = await clean_text(text)
            return cleaned_text
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return None

async def call_serper_api(web_search_request: WebSearchRequest):
    url = "https://google.serper.dev/search"
    payload = json.dumps({
        "q": web_search_request.query,
        "num": web_search_request.num_results
        })
    headers = {'X-API-KEY': web_search_request.serper_api_key, 'Content-Type': 'application/json'}
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, data=payload) as response:
            data = await response.json()
            return data

def parse_embedding(embedding_str: str) -> List[float]:
    return list(map(float, embedding_str.split(',')))

async def create_embeddings_for_texts(chunks: List[str], openai_api_key: str) -> List[List[float]]:
    ''' 
    Create embeddings for a list of texts.

    Args:
        chunks (list[str]): List of text chunks to create embeddings for

    Returns:
        embeddings (list[list[float]]): List of embeddings for each chunk
    '''

    logger.debug(f"Creating embeddings for {len(chunks)} chunks")

    block_size = 10
    embeddings: List[str] = []  # converted from list of ints

    for i in range(0, len(chunks), block_size):
        block = chunks[i:i+block_size]
        response = await openai.Embedding.acreate(
            api_key=openai_api_key,
            input=block,  # Send a block of texts 
            model="text-embedding-ada-002"
        )

        # Sort the embedding objects by the index
        sorted_embed_objs = sorted(response['data'], key=lambda x: x["index"])
        
        for embedding_object in sorted_embed_objs:
            # Convert embedding to string
            embedding_as_string = ','.join(map(str, embedding_object['embedding']))
            # Append to list of embeddings
            embeddings.append(embedding_as_string)

    logger.debug(f"Created {len(embeddings)} embeddings for chunks")

    return embeddings