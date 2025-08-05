import os
from constructs import Construct
from aws_cdk import (
    Duration,
    RemovalPolicy,
    Stack,
    aws_lambda as _lambda,
    aws_apigateway as _apigateway,
    aws_sqs as _sqs
)
from aws_cdk.aws_iam import (
    ManagedPolicy
)


class WhatToCookStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # create infra
        self.create_infra()

    def create_infra(self):
        # create sqs
        queue = self.create_sqs()

        # create send message to sqs rest api
        self.create_send_message_to_sqs_rest_api(queue)

        # create agent response rest api
        self.create_agent_response_rest_api(queue)

    def create_sqs(self):
        queue = _sqs.Queue(
            self,
            "WhatToCookSqsQueue",
            queue_name="what-to-cook-sqs-queue",
            visibility_timeout=Duration.seconds(30),
            removal_policy=RemovalPolicy.DESTROY
        )
        return queue

    def create_send_message_to_sqs_rest_api(self, queue):
        # create lambda layer
        lambda_layer = _lambda.LayerVersion(
            self,
            "WhatToCookSqsLambdaLayer",
            code=_lambda.Code.from_asset("lambda/layer"),
            compatible_runtimes=[_lambda.Runtime.PYTHON_3_12],
            compatible_architectures=[_lambda.Architecture.ARM_64],
            removal_policy=RemovalPolicy.RETAIN,
        )

        # create lambda
        lambda_func = _lambda.Function(
            self,
            "WhatToCookSqsLambdaFunc",
            function_name="what-to-cook-sqs-lambda",
            runtime=_lambda.Runtime.PYTHON_3_12,
            memory_size=1000,
            code=_lambda.Code.from_asset("lambda"),
            handler="send_message_to_sqs.handler",
            architecture=_lambda.Architecture.ARM_64,
            timeout=Duration.minutes(5),
            tracing=_lambda.Tracing.ACTIVE,
            layers=[lambda_layer],
            environment={
                "PYTHON_ENV": os.getenv("PYTHON_ENV"),
                "SQS_QUEUE_URL": queue.queue_url
            },
        )

        lambda_func.role.add_managed_policy(
            ManagedPolicy.from_aws_managed_policy_name("AdministratorAccess"),
        )

        # add send message to lambda func
        queue.grant_send_messages(lambda_func)

        # create api gateway
        api_gateway = _apigateway.LambdaRestApi(
            self,
            "WhatToCookSqsApiGateway",
            handler=lambda_func,
            default_cors_preflight_options=_apigateway.CorsOptions(
                allow_origins=_apigateway.Cors.ALL_ORIGINS,
                allow_methods=_apigateway.Cors.ALL_METHODS
            ),
            proxy=False,
            integration_options=_apigateway.LambdaIntegrationOptions(
                timeout=Duration.seconds(29)
            ),
            deploy_options=_apigateway.StageOptions(
                data_trace_enabled=True,
                tracing_enabled=True,
                metrics_enabled=True
            )
        )

        api = api_gateway.root.add_resource("what-to-cook")
        send_message_to_sqs = api.add_resource("send-message-to-sqs")
        # POST /what-to-cook/send-message-to-sqs
        send_message_to_sqs.add_method("POST")

    def create_agent_response_rest_api(self, queue):
        # create lambda layer
        lambda_layer = _lambda.LayerVersion(
            self,
            "WhatToCookLambdaLayer",
            code=_lambda.Code.from_asset("lambda/layer"),
            compatible_runtimes=[_lambda.Runtime.PYTHON_3_12],
            compatible_architectures=[_lambda.Architecture.ARM_64],
            removal_policy=RemovalPolicy.RETAIN,
        )

        # create lambda
        lambda_func = _lambda.Function(
            self,
            "WhatToCookLambdaFunc",
            function_name="what-to-cook",
            runtime=_lambda.Runtime.PYTHON_3_12,
            memory_size=1000,
            code=_lambda.Code.from_asset("lambda"),
            handler="what_to_cook.handler",
            architecture=_lambda.Architecture.ARM_64,
            timeout=Duration.minutes(5),
            tracing=_lambda.Tracing.ACTIVE,
            layers=[lambda_layer],
            environment={
                "PYTHON_ENV": os.getenv("PYTHON_ENV"),
                "SQS_QUEUE_URL": queue.queue_url
            },
        )

        lambda_func.role.add_managed_policy(
            ManagedPolicy.from_aws_managed_policy_name("AdministratorAccess"),
        )

        # create api gateway
        api_gateway = _apigateway.LambdaRestApi(
            self,
            "WhatToCookApiGateway",
            handler=lambda_func,
            default_cors_preflight_options=_apigateway.CorsOptions(
                allow_origins=_apigateway.Cors.ALL_ORIGINS,
                allow_methods=_apigateway.Cors.ALL_METHODS
            ),
            proxy=False,
            integration_options=_apigateway.LambdaIntegrationOptions(
                timeout=Duration.seconds(29)
            ),
            deploy_options=_apigateway.StageOptions(
                data_trace_enabled=True,
                tracing_enabled=True,
                metrics_enabled=True
            )
        )

        api = api_gateway.root.add_resource("what-to-cook")
        api.add_method("GET")  # GET /what-to-cook
