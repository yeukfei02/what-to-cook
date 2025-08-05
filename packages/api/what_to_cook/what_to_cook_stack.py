import os
from constructs import Construct
from aws_cdk import (
    Duration,
    RemovalPolicy,
    Stack,
    aws_lambda as _lambda,
    aws_apigateway as _apigateway
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
        # create lambda layer
        lambda_layer = self.create_lambda_layer()

        # create lambda
        lambda_func = self.create_lambda(lambda_layer)

        # create api gateway
        self.create_api_gateway(lambda_func)

    def create_lambda_layer(self):
        lambda_layer = _lambda.LayerVersion(
            self,
            "WhatToCookLambdaLayer",
            code=_lambda.Code.from_asset("lambda/layer"),
            compatible_runtimes=[_lambda.Runtime.PYTHON_3_12],
            compatible_architectures=[_lambda.Architecture.ARM_64],
            removal_policy=RemovalPolicy.RETAIN,
        )
        return lambda_layer

    def create_lambda(self, lambda_layer):
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
            },
        )

        lambda_func.role.add_managed_policy(
            ManagedPolicy.from_aws_managed_policy_name("AdministratorAccess"),
        )

        return lambda_func

    def create_api_gateway(self, lambda_func):
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
