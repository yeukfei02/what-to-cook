#!/usr/bin/env python3
from dotenv import load_dotenv
import aws_cdk as cdk
from what_to_cook.what_to_cook_stack import WhatToCookStack
from helper.helper import get_env

load_dotenv()

app = cdk.App()

env = get_env()

WhatToCookStack(
    app,
    "WhatToCookStack",
    stack_name="what-to-cook-stack",
    env=env
)

app.synth()
