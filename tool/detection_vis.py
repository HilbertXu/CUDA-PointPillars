import csv

import numpy
import numpy as np  
from tools.visual_utils.open3d_vis_utils import draw_scenes

pt_file = "../seq/000009.bin"
det_file = "../resutls/000009.csv" 

points = np.fromfile(pt_file, dtype=np.float32).reshape(-1,4)
print(points.shape)


box_list = []
score_list = []
id_list = []

with open(det_file) as f:
    lines = f.readlines()
    for i in range(1, len(lines)):
        line = lines[i]
        data = line.split(";")
        box = np.asarray([float(data[0]), float(data[1]), float(data[2]), float(data[3]), float(data[4]), float(data[5]), float(data[6])]).reshape(1,7)
        cls_id = np.asarray([int(data[-2])]).reshape(1)
        score = np.asarray([float(data[-1])]).reshape(1)
        print(box)

        box_list.append(box)
        id_list.append(cls_id)
        score_list.append(score)


boxes = np.concatenate(box_list, axis=0)
scores = np.concatenate(score_list, axis=0)
cls_ids = np.concatenate(id_list, axis=0)

mask = scores > 0.1
boxes = boxes[mask]
scores = scores[mask]
cls_ids = cls_ids[mask]

print(boxes.shape)
print(scores.shape)
print(cls_ids.shape)

draw_scenes(points, ref_boxes=boxes, ref_labels=cls_ids, ref_scores=scores)
# draw_scenes(points)
        


