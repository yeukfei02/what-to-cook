from fastapi import APIRouter, Request
from controller.send_message_to_sqs_controller import send_message_to_sqs_controller
from controller.what_to_cook_controller import what_to_cook_controller

data_router = APIRouter()


@data_router.post("/what-to-cook/send-message-to-sqs")
def send_message_to_sqs(request: Request):
    return send_message_to_sqs_controller(request)


@data_router.get("/what-to-cook")
def what_to_cook(request: Request):
    return what_to_cook_controller(request)
