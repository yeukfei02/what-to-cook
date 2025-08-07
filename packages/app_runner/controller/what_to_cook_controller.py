from fastapi import Request, status
from fastapi.responses import JSONResponse


async def what_to_cook_controller(request: Request):
    data = {
        "data": "",
    }

    data = {
        "data": "",
    }

    response = JSONResponse(status_code=status.HTTP_200_OK, content=data)
    return response
