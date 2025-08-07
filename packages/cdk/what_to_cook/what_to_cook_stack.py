import os
from constructs import Construct
from aws_cdk import (
    Duration,
    RemovalPolicy,
    Stack,
    aws_lambda as _lambda,
    aws_apigateway as _apigateway,
    aws_sqs as _sqs,
    aws_ecr as _ecr,
    aws_iam as _iam,
    aws_apprunner as _apprunner,
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
        # create lambda and api gateway
        self.create_lambda_and_api_gateway()

        # create app runner
        # self.create_app_runner()

    def create_lambda_and_api_gateway(self):
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

    def create_app_runner(self):
        # create ecr repository
        ecr_repo = _ecr.Repository(
            self, "WhatToCookEcrRepo", repository_name="what-to-cook-ecr")

        # create iam role
        iam_role = _iam.Role(
            self,
            "WhatToCookAppRunnerIamRole",
            role_name="what-to-cook-app-runner-iam-role",
            assumed_by=_iam.ServicePrincipal("build.apprunner.amazonaws.com")
        )

        # create iam policy statement
        policy_statement = _iam.PolicyStatement(
            actions=["ecr:*"],
            resources=["*"]
        )

        # create iam policy
        policy = _iam.Policy(
            self,
            "WhatToCookAppRunnerIamPolicy",
            policy_name="what-to-cook-app-runner-iam-policy",
            statements=[policy_statement]
        )

        # attach iam policy to iam role
        policy.attach_to_role(iam_role)

        # create app runner
        image_tag = "latest"

        _apprunner.CfnService(
            self,
            "WhatToCookAppRunnerService",
            service_name="what-to-cook-app-runner-service",
            source_configuration=_apprunner.CfnService.SourceConfigurationProperty(
                auto_deployments_enabled=True,
                authentication_configuration=_apprunner.CfnService.AuthenticationConfigurationProperty(
                    access_role_arn=iam_role.role_arn
                ),
                image_repository=_apprunner.CfnService.ImageRepositoryProperty(
                    image_identifier=f"{ecr_repo.repository_uri}:{image_tag}",
                    image_repository_type="ECR",
                    image_configuration=_apprunner.CfnService.ImageConfigurationProperty(
                                          port="80"
                                          # start_command="nginx -g daemon off;"
                    )
                )
            ),
            instance_configuration=_apprunner.CfnService.InstanceConfigurationProperty(
                cpu="1 vCPU",
                memory="2 GB"
            )
        )
