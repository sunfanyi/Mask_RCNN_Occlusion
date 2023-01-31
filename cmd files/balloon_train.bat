call conda activate mrcnn
cd ../samples/balloon
python balloon.py train --dataset=../../datasets/balloon/ --weights=coco
call conda deactivate

PAUSE