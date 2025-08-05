import requests


def what_to_cook_api(user_input_value):
    result = None

    try:
        root_url = "https://89s16vof83.execute-api.us-east-1.amazonaws.com/prod"

        params = {
            "input": user_input_value,
        }

        response = requests.get(
            f"{root_url}/what-to-cook", params=params)
        print(f"response = {response}")

        if response:
            response_json = response.json()
            print(f"response_json = {response_json}")

            if response_json:
                data = response_json["data"]
                if data:
                    result = data
    except Exception as e:
        print(f"what_to_cook_api error = {e}")

    return result
