from .logger_config import get_logger
logger = get_logger(__name__)

import logging
import asyncio
import aiohttp
import json

from .helpers import file_parsing_wrapper, create_embeddings_for_texts, \
    parse_embedding, call_serper_api, fetch_page, fetch_data_from_bubble
from .models import VectorSimilarity, WebSearchRequest, SiteReadRequest, FileParsingRequest

from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from sklearn.metrics.pairwise import cosine_similarity
import heapq

router = APIRouter()

@router.get("/data/ping")
@router.get("/ping")
async def ping():
    return {
        "message": "pong",
        "status_code": 200
    }

@router.post("/data/read_websites")
@router.post("/read_websites")
async def read_websites(
    site_read_request: SiteReadRequest):
    ''' Take a list of URLs and return the contents of the page, formatted
    as plain text.
    '''
    
    try:
        urls = site_read_request.urls.split(',')
        logger.debug(f"{urls=}")
        # add https:// to any URLs that don't have it
        stripped_urls = [url.strip() for url in urls]
        urls = [url if url.startswith("http") else f"https://{url}" for url in stripped_urls]
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for url in urls:
                tasks.append(fetch_page(session, url))
            
            pages_content = await asyncio.gather(*tasks)
            
            # Prepare result
            results = []
            for url, content in zip(urls, pages_content):
                if content is not None:
                    if site_read_request.include_source_url:
                        result_string = f"source: {url}\n\n {content}"
                    else:
                        result_string = content
                    results.append(result_string)
                    
            return JSONResponse(
                status_code=200,
                content={
                    "results": results,
                    "message": "Successfully fetched page contents",
                    "status_code": 200,
            })
    except Exception as e:
        logger.error("Error fetching page contents")
        logger.error(e, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "message": "Error fetching page contents",
                "status_code": 500 # including in body b/c Bubble doesn't get the response status code 
            })

@router.post("/data/web_search")
@router.post("/web_search")
async def fetch_page_contents(
    web_search_request: WebSearchRequest
    ):
    ''' Accept a WebSearchRequest and return the contents of the top N search results.

    Args:
        web_search_request (WebSearchRequest): WebSearchRequest object
            web_search_request.query (str): Query to search for
            web_search_request.serper_api_key (str): Serper API key to use for search
            web_search_request.num_results (int): Number of results to return
    
    Returns:
        results (list[str]): List of JSON strings containing the source URL and content

    Note - This is tied a bit too closely to Serper API. We should probably
    abstract this out a bit more.
    '''
    
    try:
        payload = await call_serper_api(web_search_request)
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for entry in payload["organic"]:
                tasks.append(fetch_page(session, entry['link']))
            
            pages_content = await asyncio.gather(*tasks)
            
            # Prepare result
            results = []
            for url, content in zip([entry['link'] for entry in payload["organic"]], pages_content):
                if content is not None:

                    if web_search_request.include_source_url:
                        result_string = f"source: {url}\n\n {content}"
                    else:
                        result_string = content

                    results.append(result_string)

            
            return JSONResponse(
                status_code=200,
                content= {
                    "results": results,
                    "message": "Web search results fetched successfully",
                    "status_code": 200,
            })

    except Exception as e:
        logger.error("Error fetching web search results")
        logger.error(e, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "message": "Error fetching web search results",
                "status_code": 500 # including in body b/c Bubble doesn't get the response status code 
            })



@router.post("/data/extract_text_and_embeddings_from_file")
@router.post("/extract_text_and_embeddings_from_file")
async def extract_text_and_embeddings_from_file(
    background_tasks: BackgroundTasks,
    file_parsing_request: FileParsingRequest
    ):
    '''  Download and parse a file.

    Can return synchronous or asynchronous results, depending on the
    wf_callback_url or data_api_url

    Args:
        file_parsing_request (FileParsingRequest): FileParsingRequest object
            file_parsing_request.file_url (str): URL of file to parse
            file_parsing_request.openai_api_key (str): OpenAI API key to use for embeddings
            file_parsing_request.create_embeddings (bool): Whether to create embeddings for the text chunks
            file_parsing_request.custom_plugin_url (str): URL of custom plugin to use for parsing
            file_parsing_request.wf_api_url (str): URL to POST results to
            file_parsing_request.data_api_url (str): URL of data API to POST results to
            file_parsing_request.chunk_data_type (str): Data type of the chunks
            file_parsing_request.chunk_text_field (str): Field name for the text chunks
            file_parsing_request.chunk_embeddings_field (str): Field name for the embeddings

    Returns:
        result_data: JSON object containing the results
            result_data.chunks (list[str]): List of text chunks
            result_data.embeddings (list[str]): List of embeddings
            result_data.message (str): Message indicating success or failure
    '''

    try:
        logger.info("Request to POST /extract_text_and_embeddings_from_file")
        
        if file_parsing_request.data_api_url or file_parsing_request.wf_api_url:
            # Run the processing asynchronously
            background_tasks.add_task(
                file_parsing_wrapper,
                file_parsing_request
                )
            return JSONResponse(
                status_code=200,
                content={
                    "message": "File processing started",
                    "status_code": 200
                })
        else:
            # Run the processing synchronously
            result_data = await file_parsing_wrapper(file_parsing_request)
            return JSONResponse(
                status_code=200,
                # Note - 
                content= {
                    "chunks": result_data["texts"],
                    "embeddings": result_data["embeddings"],
                    "message": "File processed successfully",
                    "status_code": 200
            })
    except Exception as e:
        logger.error("Error generating embeddings")
        logger.error(e, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "message": "Error generating embeddings",
                "status_code": 500
            })

@router.post("/data/vector_similarity")
@router.post("/vector_similarity")
async def vector_similarity_endpoint(request: VectorSimilarity):
    ''' Accept a VectorSimilarity request and return the top 'n' matches.

    Args:
        request (VectorSimilarity): VectorSimilarity request object
    '''

    try:
        logger.info("Request to POST /vector_similarity")
        logger.info(request)

        # Vectorize the incoming text
        embedded_query = await create_embeddings_for_texts(chunks=[request.input_text], openai_api_key=request.openai_api_key)
        embedded_query_to_vector = parse_embedding(embedded_query[0])

        # Fetch chunks from bubble
        records =  await fetch_data_from_bubble(request)

        # Extract embeddings list
        embeddings = [record[request.chunk_embeddings_field] for record in records]

        # Convert each embedding, stored as a string, to a list of floats
        parsed_embeddings = [parse_embedding(embed_str) for embed_str in embeddings]

        # Calculate cosine similarity
        similarity_scores = cosine_similarity([embedded_query_to_vector], parsed_embeddings)[0]

        # Sort the results
        sorted_records = sorted(zip(records, similarity_scores), key=lambda x: x[1], reverse=True)

        # Return the top 'n' results
        matching_texts = [record[request.chunk_text_field] for record,score in sorted_records][:request.return_count]
        
        return JSONResponse(
            status_code=200,
            content={
                "matching_texts": matching_texts,
                "message": "Vector similarity calculated successfully",
                "status_code": 200,
        })

    except Exception as e:
        logger.error("Error calculating vector similarity")
        logger.error(e, exc_info=True)
        return JSONResponse(
            status_code=500,
            content= {
                "message": "Error calculating vector similarity",
                "status_code": 500,
                "error": str(e)
        })