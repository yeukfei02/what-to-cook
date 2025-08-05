import requests


def send_message_to_sqs_api(user_input_value):
    result = None

    try:
        root_url = "https://g609uq5jnj.execute-api.us-east-1.amazonaws.com/prod"

        body_json = {
            "input": user_input_value,
        }

        response = requests.post(
            f"{root_url}/what-to-cook/send-message-to-sqs", json=body_json)
        print(f"response = {response}")

        if response:
            response_json = response.json()
            print(f"response_json = {response_json}")

            if response_json:
                data = response_json["data"]
                if data:
                    result = data
    except Exception as e:
        print(f"send_message_to_sqs_api error = {e}")

    return result
