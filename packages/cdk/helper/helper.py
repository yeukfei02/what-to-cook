import os
import aws_cdk as cdk


def get_env():
    account_id = os.getenv("ACCOUNT_ID")
    env = cdk.Environment(account=account_id, region="us-east-1")

    return env
