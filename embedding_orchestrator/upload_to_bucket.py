from google.cloud import storage
from dotenv import load_dotenv
import os

load_dotenv()

def upload_string_to_bucket(bucket_name: str, content: str, destination_blob_name: str):
    """
    Upload a string directly to a GCS bucket.

    :param bucket_name: Name of your GCS bucket
    :param content: String content to upload
    :param destination_blob_name: File name in the bucket
    """
    # Initialize client
    client = storage.Client()
    
    # Get the bucket
    bucket = client.bucket(bucket_name)
    
    # Create a blob object
    blob = bucket.blob(destination_blob_name)
    
    # Upload string
    blob.upload_from_string(content)
    
    print(f"Uploaded string to {destination_blob_name} in bucket {bucket_name}.")
