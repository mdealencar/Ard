#!/bin/bash
pytest --cov=ard test/system

rm -rf test/unit/layout/problem*_out

#
