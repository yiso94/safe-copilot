#!/bin/bash

uv sync
uv pip install -r requirements.txt --index-strategy unsafe-best-match
