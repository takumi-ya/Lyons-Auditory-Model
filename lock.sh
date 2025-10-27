#!/bin/bash

pip install -r requirements.txt --root-user-action=ignore

if [ $? -ne 0 ]; then
    echo "ERROR:pip install failed" >&2
    exit 1
fi

if pip check; then
    pip freeze > requirements.lock

    diff requirements.txt requirements.lock || true
else
    echo "ERROR:pip check failed" >&2
    exit 1
fi
