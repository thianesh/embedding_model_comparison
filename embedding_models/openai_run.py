import os
# Update this with your self-hosted endpoint
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "http://localhost:6006/v1/traces"
from dotenv import load_dotenv
load_dotenv()

from phoenix.otel import register

# configure the Phoenix tracer
tracer_provider = register(
  project_name="my-llm-app", # Default is 'default'
  auto_instrument=True, # See 'Trace all calls made to a library' below
)
tracer = tracer_provider.get_tracer(__name__)

# Add OpenAI API Key
import os
import openai

client = openai.OpenAI()
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Write a haiku."}],
)
print(response.choices[0].message.content)