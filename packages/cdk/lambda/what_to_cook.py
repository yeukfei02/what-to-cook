import os
import json
import boto3


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
        agent_response = get_message_from_sqs_queue()

        response = {
            "statusCode": 200,
            "body": json.dumps({
                "data": agent_response
            })
        }

    print(f"response = {response}")
    return response


def get_message_from_sqs_queue():
    result = ""

    try:
        sqs = boto3.client("sqs")

        queue_url = os.getenv("SQS_QUEUE_URL")

        response = sqs.receive_message(
            QueueUrl=queue_url,
            MaxNumberOfMessages=10,
            WaitTimeSeconds=10,
            VisibilityTimeout=30
        )

        if "Messages" in response:
            for message in response["Messages"]:
                message_id = message["MessageId"]
                message_body = message["Body"]
                receipt_handle = message["ReceiptHandle"]
                print(f"message_id = {message_id}")
                print(f"message_body = {message_body}")
                print(f"receipt_handle = {receipt_handle}")

                if message_body:
                    result += message_body

                sqs.delete_message(
                    QueueUrl=queue_url,
                    ReceiptHandle=receipt_handle
                )
                print(f"delete_message = {message_id}")
    except Exception as e:
        print(f"get_message_from_sqs_queue error = {e}")

    return result
