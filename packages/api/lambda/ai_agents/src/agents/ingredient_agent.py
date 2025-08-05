from strands import Agent
from strands.models import BedrockModel


bedrock_model = BedrockModel(
    model_id="us.meta.llama4-scout-17b-instruct-v1:0",
    region_name="us-east-1",
    temperature=0.5,
)

ingredient_agent = Agent(
    model=bedrock_model,
    system_prompt=(
        "You are a helpful ingredient assistant that can suggest ingredients based on user preferences. "
        "You can provide detailed information about ingredients, their uses, and how they can be incorporated into recipes."
        "Use your knowledge to provide accurate and helpful responses."
    ),
    tools=[],
)
