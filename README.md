# what-to-cook

what-to-cook

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
$ pip install --platform manylinux2014_aarch64 --target=./ --implementation cp --python-version 3.12 --only-binary=:all: --upgrade pydantic
```
