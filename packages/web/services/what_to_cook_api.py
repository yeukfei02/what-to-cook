import json
import requests


def what_to_cook_api(user_input_value):
    result = None

    try:
        root_url = "http://localhost:8080"

        payload = json.dumps({
            "prompt": user_input_value
        })

        response = requests.post(
            f"{root_url}/invocations",
            data=payload
        )
        print(f"response = {response}")

        if response:
            response_json = response.json()
            print(f"response_json = {response_json}")

            if response_json:
                data = response_json["data"]
                if data:
                    result = data.replace("\n", "<br>")
    except Exception as e:
        print(f"what_to_cook_api error = {e}")

    return result
