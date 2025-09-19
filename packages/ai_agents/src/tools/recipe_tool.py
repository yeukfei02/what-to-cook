from strands import tool
from src.agents.recipe_agent import recipe_agent


@tool
def recipe_assistant(query: str) -> str:
    result = None

    try:
        recipe_agent_response = recipe_agent(query)

        result = recipe_agent_response
    except Exception as e:
        print(f"recipe_assistant error = {e}")

    return str(result)
