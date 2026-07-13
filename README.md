# Smart Sprayer: Crop and Weed Detection

## Problem
Weeds reduce crop yield. Spraying pesticides everywhere wastes chemicals and contaminates crops.

## Aim
Develop a YOLOv8 model that detects `crop` vs `weed` and enables precision spraying only on weeds.

## Dataset
- 1300 images, 512x512
- 2 Classes: crop, weed
- [Download Dataset](YOUR_DRIVE_LINK_HERE)

## How to Run
1. `pip install -r requirements.txt`
2. `python src/train.py`
3. `python src/predict.py --img path/to/image.jpg`

## Results
Add your mAP and prediction screenshots here
