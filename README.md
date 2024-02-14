# Coarse segmentation

### Step 1: Create dataset
Download the zip files from [WIDER Face dataset](http://shuoyang1213.me/WIDERFACE/).
Then see the section `Create dataset - original images and their masks' in create_masks.ipynb to create 2 folders for original images (resized and padded) and their masks.

### Step 2: Train
```
python train_mobilenetv3.py
```
(Validation/testing to be added)

### Step 3: Infer
```
python infer.py
```
