# move_files.py - FINAL FIX
import os
import shutil
import random

TRAIN_IMG  = r"dataset\train_images"
TRAIN_MASK = r"dataset\train_masks"
VAL_IMG    = r"dataset\val_images"
VAL_MASK   = r"dataset\val_masks"

all_images = os.listdir(TRAIN_IMG)
matched = []
for fname in all_images:
    base       = os.path.splitext(fname)[0]
    mask_fname = base + "_lab.png"
    mask_path  = os.path.join(TRAIN_MASK, mask_fname)
    if os.path.exists(mask_path):
        matched.append(fname)
    else:
        print("Skipped (no mask): " + fname)

print("Matched pairs: " + str(len(matched)))

random.seed(42)
val_files = random.sample(matched, int(len(matched) * 0.2))

for fname in val_files:
    base       = os.path.splitext(fname)[0]
    mask_fname = base + "_lab.png"
    shutil.move(os.path.join(TRAIN_IMG,  fname),      os.path.join(VAL_IMG,  fname))
    shutil.move(os.path.join(TRAIN_MASK, mask_fname), os.path.join(VAL_MASK, mask_fname))
    print("Moved: " + fname + " + " + mask_fname)

print("Done. Moved " + str(len(val_files)) + " pairs to val folders.")
print("Train images: " + str(len(os.listdir(TRAIN_IMG))))
print("Train masks:  " + str(len(os.listdir(TRAIN_MASK))))
print("Val images:   " + str(len(os.listdir(VAL_IMG))))
print("Val masks:    " + str(len(os.listdir(VAL_MASK))))