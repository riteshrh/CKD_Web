import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

# Load env variables
load_dotenv('.env')

# Load static context
try:
    with open('context.txt', 'r') as file:
        general_context = file.read().strip()
except FileNotFoundError:
    general_context = "No context loaded."

# Init Azure OpenAI
llm = AzureChatOpenAI(
    deployment_name=os.environ['MODEL'],
    openai_api_version=os.environ['API_VERSION'],
    openai_api_key=os.environ['OPENAI_API_KEY'],
    azure_endpoint=os.environ['OPENAI_API_BASE'],
    openai_organization=os.environ['OPENAI_ORGANIZATION']
)