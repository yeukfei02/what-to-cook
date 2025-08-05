# what-to-cook

what to cook is a multi llm agent for user to find what to eat or what to cook by searching different ingredients.

## Tech Stack

- python
- strands agent
- aws cdk
- aws sqs
- aws api gateway
- aws lambda
- nicegui

ai agents: `packages/api/lambda/ai_agents`

api/cdk: `packages/api`

web: `packages/web`

## Requirement

- install python (v3.12)

```zsh
// create virtualenv
$ python -m venv .venv

// activate virtualenv
$ source .venv/bin/activate
or
$ sh .venv/bin/activate

// install dependencies
$ pip install -r requirements.txt

// run in local
$ python app.py
```

```zsh
// copy .env file
$ cp .env.sample .env

// fix pydantic lib in lambda layer
$ cd packages/api/lambda/layer/python 
$ pip install --platform manylinux2014_aarch64 --target=./ --implementation cp --python-version 3.12 --only-binary=:all: --upgrade pydantic
```
