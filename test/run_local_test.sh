#!/bin/bash
# pytest --cov=windard test/unit/cost
pytest --cov=windard test

rm -rf test/unit/layout/problem*_out

#
