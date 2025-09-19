import asyncio
from nicegui import ui
from services.what_to_cook_api import what_to_cook_api


def handle_user_input(value):
    print(f"value = {value}")


async def handle_submit_button_click(user_input_value):
    if user_input_value:
        ui.notify('Please wait, processing your request...')

        agent_response_html.content = "Waiting for response..."

        # await asyncio.sleep(3)

        data = what_to_cook_api(user_input_value)
        print(f"data = {data}")

        if data:
            agent_response_html.content = data
        else:
            agent_response_html.content = 'No response from the api. Please try again.'


with ui.column().classes('w-full h-screen flex justify-center items-center'):
    ui.label('What to Cook?').classes('text-3xl font-bold mb-3')

    user_input = ui.textarea(
        # label='Enter your ingredients',
        placeholder='Enter your ingredients here',
        on_change=lambda e: handle_user_input(e.value),
        value="What can I cook with egg, tomato, chicken, rice, and vegetables? Based on these ingredients, suggest a food and how I can prepare it."
    ).props('clearable').classes('w-3/6 border rounded-lg px-3 text-lg')

    submit_button = ui.button('Submit', on_click=lambda: handle_submit_button_click(
        user_input.value)).classes('w-2/12 my-4')

    with ui.column().classes('w-3/6 p-3'):
        ui.label(
            'Agent Response:').classes('text-lg font-bold')
        agent_response_html = ui.html('Waiting for response...')

ui.run(title="What to Cook?", port=9000)
