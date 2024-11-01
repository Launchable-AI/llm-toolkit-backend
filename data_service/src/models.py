from pydantic import BaseModel
from typing import List, Any
from fastapi import UploadFile, File

class VectorSimilarity(BaseModel):
    input_text: str
    chunk_data_type: str | None = None
    chunk_embeddings_field: str | None = None
    chunk_text_field: str | None = None
    chunk_unique_ids: List[str] | None = None
    constraint_key: str | None = None
    constraint_type: str | None = None
    constraint_value: Any | None = None
    return_count: int = 5
    openai_api_key: str | None = None
    data_api_url: str | None = None
    bubble_api_key: str | None = None
    embedding_model: str | None = "text-embedding-3-large"
    version: int | None = None

class WebSearchRequest(BaseModel):
    query: str
    serper_api_key: str
    num_results: int = 5
    include_source_url: bool = True

class SiteReadRequest(BaseModel):
    urls: str
    include_source_url: bool = True

class FileParsingRequest(BaseModel):
    file_url: str
    openai_api_key: str | None = None
    bubble_api_key: str | None = None
    create_embeddings: bool = False
    custom_plugin_url: str | None = None
    wf_api_url: str | None = None
    data_api_url: str | None = None
    chunk_data_type: str | None = None
    chunk_text_field: str | None = None
    chunk_embeddings_field: str | None = None
    chunk_page_num_field: str | None = None
    reference_field: str | None = None
    reference_id: str | None = None
    embedding_model: str | None = "text-embedding-3-large"
    partition_strategy: str | None = "fast"

class VideoProcessingRequest(BaseModel):
    video_url: str
    video_id: str
    bubble_webhook_url: str
    datatype: str = None
    processor_lambda_arn: str
    count: int = None