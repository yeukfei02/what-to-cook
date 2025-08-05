from dotenv import load_dotenv
from strands import Agent
from strands.models import BedrockModel
from ai_agents.src.tools.chef_tool import chef_assistant
from ai_agents.src.tools.ingredient_tool import ingredient_assistant
from ai_agents.src.tools.recipe_tool import recipe_assistant

load_dotenv()

bedrock_model = BedrockModel(
    model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    region_name="us-east-1",
    temperature=0.5,
)

orchestrator_agent = Agent(
    model=bedrock_model,
    system_prompt=(
        "You are a helpful assistant that can suggest recipes and ingredients based on user preferences. "
        "You can provide detailed instructions on how to prepare dishes and suggest ingredients based on user preferences. "
        "Use your knowledge to provide accurate and helpful responses."
    ),
    callback_handler=None,
    tools=[
        chef_assistant,
        ingredient_assistant,
        recipe_assistant
    ]
)

# query = "What can I cook with egg, tomato, chicken, rice, and vegetables? Based on these ingredients, suggest a food and how I can prepare it."
# response = orchestrator_agent(query)
# print(f"orchestrator response = {response}\n")
