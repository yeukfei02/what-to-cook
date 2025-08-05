from nicegui import ui
from services.what_to_cook_api import what_to_cook_api


def handle_user_input(value):
    print(f"value = {value}")


def handle_submit_button_click(user_input_value):
    if user_input_value:
        ui.notify('Please wait, processing your request...')

        data = what_to_cook_api(user_input_value)
        print(f"data = {data}")

        if data:
            agent_response.set_text(data)
        else:
            agent_response.set_text(
                'No response from the api. Please try again.')


with ui.column().classes('w-full h-screen flex justify-center items-center'):
    ui.label('What to Cook?').classes('text-3xl font-bold mb-3')

    user_input = ui.textarea(
        # label='Enter your ingredients',
        placeholder='Enter your ingredients here',
        on_change=lambda e: handle_user_input(e.value),
    ).props('clearable').classes('w-3/6 border rounded-lg px-3 text-lg')

    ui.button('Submit', on_click=lambda: handle_submit_button_click(
        user_input.value)).classes('w-2/12 my-4')

    ui.label('Agent Response:').classes('text-lg font-bold')

    agent_response = ui.label('Waiting for response...')

ui.run(title="What to Cook?")
