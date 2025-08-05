import json
from ai_agents.app import orchestrator_agent


def handler(event, context):
    print(f"event = {event}")
    print(f"context = {context}")

    response = {
        "statusCode": 200,
        "body": json.dumps({
            "data": ""
        })
    }

    if event:
        if event["queryStringParameters"]:
            input = event["queryStringParameters"]["input"]
            if input:
                agent_response = orchestrator_agent(input)

                response = {
                    "statusCode": 200,
                    "body": json.dumps({
                        "data": str(agent_response)
                    })
                }

    print(f"response = {response}")
    return response
