#!/bin/bash
# pytest --cov=ard test/unit/cost
pytest --cov=ard test

rm -rf test/unit/layout/problem*_out

#
