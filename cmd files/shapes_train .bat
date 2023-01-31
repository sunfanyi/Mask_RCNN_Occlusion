#!/bin/bash
call conda activate mrcnn
cd ../samples/shapes
python shapes.py train --weights=coco
call conda deactivate

PAUSE