#!/bin/bash
pytest --cov=ard test/unit

rm -rf test/unit/layout/problem*_out

#
