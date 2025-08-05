from strands import Agent
from strands.models import BedrockModel


bedrock_model = BedrockModel(
    model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
    region_name="us-east-1",
    temperature=0.5,
)

chef_agent = Agent(
    model=bedrock_model,
    system_prompt=(
        "You are a helpful chef assistant that can suggest recipes and ingredients. "
        "You can provide detailed instructions on how to prepare dishes and suggest ingredients based on user preferences."
        "Use your knowledge to provide accurate and helpful responses."
    ),
    tools=[],
)
