from strands import tool
from src.agents.ingredient_agent import ingredient_agent


@tool
def ingredient_assistant(query: str) -> str:
    result = None

    try:
        ingredient_agent_response = ingredient_agent(query)

        result = ingredient_agent_response
    except Exception as e:
        print(f"ingredient_assistant error = {e}")

    return str(result)
