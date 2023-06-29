import torch
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
setup_logger(name="mask2former")
import numpy as np
import cv2
import torch
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.projects.deeplab import add_deeplab_config
coco_metadata = MetadataCatalog.get("coco_2017_val_panoptic")
from mask2former import add_maskformer2_config
from sklearn.metrics import jaccard_score
from sklearn import metrics
import os
from pathlib import Path



_OFF_WHITE = (1.0, 1.0, 240.0 / 255)

#We used 'model_final_f07440.pkl' trained using COCO. It can be downloaded at "https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/panoptic/maskformer2_swin_large_IN21k_384_bs16_100ep/model_final_f07440.pkl"
pretrained_weight_path = ""
#Directory containing all images in jpg.
img_dir = ""
#Directory containing all masks in png. Consistent with hbs dataset.
label_dir = ""
#Directory to store to masked image.
out_dir = ""


#Load baseline cfg file
cfg = get_cfg()
add_deeplab_config(cfg)
add_maskformer2_config(cfg)
#Merge pretrained cfg file with baseline file. Included in the Mask2Former file
cfg.merge_from_file("configs/coco/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml")
cfg.MODEL.WEIGHTS = pretrained_weight_path
cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = True
predictor = DefaultPredictor(cfg)

test_image = os.listdir(img_dir)

avg_iou = 0
avg_precision = 0
avg_recall = 0
avg_f1 = 0
total_img = len(test_image)
counter = 0



for cur_image in test_image:

    #WxHxC
    im = cv2.imread(img_dir + cur_image)
    #WxH
    true_mask = cv2.imread(label_dir + Path(cur_image).stem + '.png', cv2.IMREAD_UNCHANGED).astype(np.uint8)
    

    #Obtain all masks for the given image
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1], coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
    mask_color = [x / 255 for x in v.metadata.stuff_colors[0]]

    #Select the binary mask for human(class 0) and draw on the original image.
    binary_mask = (outputs["sem_seg"].argmax(0).to("cpu").numpy() == 0).astype(np.uint8)
    v.draw_binary_mask(binary_mask, color=mask_color, edge_color=_OFF_WHITE,text="human",alpha=0.8, area_threshold=None,)

    #Compute all the measurements
    avg_iou += jaccard_score(true_mask//255, binary_mask, average="micro")
    avg_precision += metrics.precision_score(true_mask//255, binary_mask, average="micro")
    avg_recall += metrics.recall_score(true_mask//255, binary_mask, average="micro")
    avg_f1 += metrics.f1_score(true_mask//255, binary_mask, average="micro")

    #Write masked image onto a directory
    cv2.imwrite(out_dir + cur_image, cv2.cvtColor(v.output.get_image(), cv2.COLOR_RGB2BGR))

#Store the scores of the predicted masks
f= open("mask2former_result.txt","w+")
f.write("mIoU " + str(avg_iou / total_img) + "\n")
f.write("Precision " + str(avg_precision / total_img) + "\n")
f.write("Recall " + str(avg_recall / total_img) + "\n")
f.write("f1 " + str(avg_f1 / total_img) + "\n")
f.close()
