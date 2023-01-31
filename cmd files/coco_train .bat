call conda activate mrcnn
cd ../samples/coco
python coco.py train --dataset=..\..\datasets\coco\ --model=coco
call conda deactivate

PAUSE