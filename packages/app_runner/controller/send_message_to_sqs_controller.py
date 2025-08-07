from fastapi import Request, status
from fastapi.responses import JSONResponse
from ai_agents.app import orchestrator_agent


async def send_message_to_sqs_controller(request: Request):
    data = {
        "data": "",
    }

    body = await request.body()
    if body:
        data = {
            "data": "",
        }

    response = JSONResponse(status_code=status.HTTP_200_OK, content=data)
    return response
