import os
import json
import boto3
from ai_agents.app import orchestrator_agent


def handler(event, context):
    print(f"event = {event}")
    print(f"context = {context}")

    if event:
        body = event.get('body')
        if body:
            body_json = json.loads(body)

            body_input = body_json.get("input")
            if body_input:
                send_message_to_sqs(body_input)


def send_message_to_sqs(body_input):
    try:
        sqs = boto3.client("sqs")

        queue_url = os.getenv("SQS_QUEUE_URL")

        message_body = orchestrator_agent(body_input)

        response = sqs.send_message(
            QueueUrl=queue_url,
            MessageBody=json.dumps(str(message_body))
        )
        print(f"response = {response}")
    except Exception as e:
        print(f"send_message_to_sqs_queue error = {e}")
