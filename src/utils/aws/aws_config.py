import boto3
import os
from dotenv import load_dotenv

load_dotenv()

def get_s3_client():
    return boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv('AWS_REGION', default='ap-southeast-1')
    )
    
def get_s3_bucket_name():
    return os.getenv('AWS_STORAGE_BUCKET_NAME')

def get_s3_config():
    return {
        'client': get_s3_client(),
        'bucket_name': get_s3_bucket_name(),
        'region': os.getenv('AWS_REGION', default='ap-southeast-1')
    }
    
def get_bedrock_client():
    return boto3.client(
        'bedrock-runtime',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID_GENAI'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY_GENAI'),
        region_name=os.getenv('AWS_REGION_GENAI', default='ap-southeast-1')
    )