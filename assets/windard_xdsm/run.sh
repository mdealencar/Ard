#!/bin/bash

python windard_xdsm.pdf
magick -quality 100 -density 300 -colorspace sRGB windard_xdsm.pdf -flatten windard_xdsm.png

