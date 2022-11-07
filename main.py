import os
import cv2
from run_model import run

existing = []
out_directory = os.fsencode('res_images')

for file in os.listdir(out_directory):
     filename = os.fsdecode(file)
     if filename.endswith(".jpeg") or filename.endswith(".png") or filename.endswith(".jpg"): 
         existing.append(filename)

in_directory = os.fsencode('input_images')

has_new = False
for file in os.listdir(in_directory):
     filename = os.fsdecode(file)
     if filename.endswith(".jpeg") or filename.endswith(".png") or filename.endswith(".jpg"): 
        ok = True
        for i in existing:
             if i == filename:
                 ok = False
                 break
        
        if ok:
            res = run(filename, to_display=False)
            print(filename)
            cv2.imwrite('res_images/' + filename, res)
        has_new = has_new or ok

if has_new:
    print('New Images saved to res_images folder')            
        