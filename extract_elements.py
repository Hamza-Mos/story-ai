#!/usr/bin/env python3
"""
extract_elements.py – split an image into separate transparent PNGs
usage: python extract_elements.py path/to/file.png
"""
import sys
import cv2
import numpy as np
from pathlib import Path

def extract(path):
    img   = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # anything >245 is considered white background here
    _, mask = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)

    # find external contours
    cntrs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out_dir  = Path(path).with_suffix('')  # same name, no extension
    out_dir.mkdir(exist_ok=True)

    for i, c in enumerate(cntrs, 1):
        x, y, w, h = cv2.boundingRect(c)
        roi = img[y:y+h, x:x+w].copy()

        # create alpha channel: keep only non-white pixels
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, alpha = cv2.threshold(roi_gray, 245, 255, cv2.THRESH_BINARY_INV)
        b, g, r = cv2.split(roi)
        rgba    = cv2.merge([b, g, r, alpha])

        cv2.imwrite(str(out_dir / f"piece_{i:02}.png"), rgba)

if __name__ == "__main__":
    extract(Path(sys.argv[1]))
