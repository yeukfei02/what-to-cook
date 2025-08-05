import json
import requests


def what_to_cook_api():
    result = None

    try:
        root_url = "https://89s16vof83.execute-api.us-east-1.amazonaws.com/prod"

        response = requests.get(f"{root_url}/what-to-cook")
        print(f"response = {response}")

        if response:
            response_json = response.json()
            print(f"response_json = {response_json}")

            if response_json:
                data = response_json["data"]
                if data:
                    decoded_text = json.loads(f"{data}")
                    result = decoded_text.replace("\n", "<br>")
    except Exception as e:
        print(f"what_to_cook_api error = {e}")

    return result
