#!/usr/bin/env python
# coding: utf-8

# # Mask R-CNN Demo
# 
# A quick intro to using the pre-trained model to detect and segment objects.

# In[1]:


import os
import sys
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
import skimage.restoration

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco
import tensorflow as tf
# tf.compat.v1.disable_eager_execution()


# get_ipython().run_line_magic('matplotlib', 'inline')

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")


# ## Configurations
# In[2]:


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()


# ## Create Model and Load Trained Weights

# In[3]:


# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)


# ## Class Names
# 
# The model classifies objects and returns class IDs, which are integer value that identify each class. Some datasets assign integer values to their classes and some don't. For example, in the MS-COCO dataset, the 'person' class is 1 and 'teddy bear' is 88. The IDs are often sequential, but not always. The COCO dataset, for example, has classes associated with class IDs 70 and 72, but not 71.
# 
# To improve consistency, and to support training on data from multiple sources at the same time, our ```Dataset``` class assigns it's own sequential integer IDs to each class. For example, if you load the COCO dataset using our ```Dataset``` class, the 'person' class would get class ID = 1 (just like COCO) and the 'teddy bear' class is 78 (different from COCO). Keep that in mind when mapping class IDs to class names.
# 
# To get the list of class names, you'd load the dataset and then use the ```class_names``` property like this.
# ```
# # Load COCO dataset
# dataset = coco.CocoDataset()
# dataset.load_coco(COCO_DIR, "train")
# dataset.prepare()
# 
# # Print class names
# print(dataset.class_names)
# ```
# 
# We don't want to require you to download the COCO dataset just to run this demo, so we're including the list of class names below. The index of the class name in the list represent its ID (first class is 0, second is 1, third is 2, ...etc.)

# In[4]:


# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


# ## Run Object Detection

###################################### Charger les images #######################################

file_names = next(os.walk(IMAGE_DIR))[2]
images = []
for file_name in file_names:
    image_path = os.path.join(IMAGE_DIR, file_name)
    image = skimage.io.imread(image_path)
    if len(image.shape) == 3 and image.shape[2] == 4:  # RGBA转RGB
        image = skimage.color.rgba2rgb(image)
    images.append((file_name, image))

labels1 = []
labels2 = []

res_dir = os.path.join(ROOT_DIR, "res")
detection_avant_dir = os.path.join(res_dir, "detection_avant")
os.makedirs(detection_avant_dir, exist_ok=True)

###################################### Définition de fonctions #######################################

#######################################
# Attaque adversal
#######################################
import skimage.util

def detection_originale(model, class_names, images, save_dir):
    """
    Cette fonction effectue la détection sur les images originales (sans perturbation).
    
    Retour:
    - labels : liste de tuples (fname, [(class_name, score, (y1,x1,y2,x2)), ...])
    """

    labels = []
    for (fname, image) in images:
        # Détection sur l'image originale
        results = model.detect([image], verbose=0)
        r = results[0]

        # Extraire les prédictions
        predictions = []
        for cid, s, box in zip(r['class_ids'], r['scores'], r['rois']):
            cname = class_names[cid]
            predictions.append((cname, float(s), tuple(box)))

        labels.append((fname, predictions))

        fig, ax = plt.subplots(1, figsize=(16, 16))
        from mrcnn import visualize
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                    class_names, r['scores'], ax=ax, title="Avant Perturbation")
        fig.savefig(os.path.join(save_dir, fname), bbox_inches='tight')
        plt.close(fig)
    return labels


def detection_avec_bruit(model, class_names, images, var=0.01):
    """
    Cette fonction effectue la détection sur les images après l'ajout d'un bruit gaussien.
    
    Retour:
    - labels : liste de tuples (fname, [(class_name, score, (y1,x1,y2,x2)), ...])
    """
    
    
    labels = []
    for (fname, image) in images:
        # Ajout du bruit gaussien
        noisy_image = skimage.util.random_noise(image, mode='gaussian', var=var)
        noisy_image = (noisy_image * 255).astype(np.uint8)

        images_bruit = os.path.join(res_dir, f"images_bruit_{var}")
        os.makedirs(images_bruit, exist_ok=True)
        skimage.io.imsave(os.path.join(images_bruit, fname), noisy_image)

        # Détection sur l'image bruyante
        results = model.detect([noisy_image], verbose=0)
        r = results[0]

        predictions = []
        for cid, s, box in zip(r['class_ids'], r['scores'], r['rois']):
            cname = class_names[cid]
            predictions.append((cname, float(s), tuple(box)))

        labels.append((fname, predictions))

        fig, ax = plt.subplots(1, figsize=(16, 16))
        from mrcnn import visualize
        visualize.display_instances(noisy_image, r['rois'], r['masks'], r['class_ids'],
                                    class_names, r['scores'], ax=ax, title="Après Perturbation (Bruit Gaussien)")

        save_dir =  os.path.join(res_dir, f"detection_apres_{var}") 
        os.makedirs(save_dir, exist_ok=True)

        fig.savefig(os.path.join(save_dir, fname), bbox_inches='tight')
        plt.close(fig)
    return labels


#######################################
# Évaluation
#######################################

def iou(boxa, boxb):
    """
    Calcule l'IoU entre deux boîtes: boxA et boxB.
    Chaque box est au format (y1, x1, y2, x2).
    """
    y1a, x1a, y2a, x2a = boxa
    y1b, x1b, y2b, x2b = boxb

    intery1 = max(y1a, y1b)
    interx1 = max(x1a, x1b)
    intery2 = min(y2a, y2b)
    interx2 = min(x2a, x2b)

    interarea = max(0, intery2 - intery1) * max(0, interx2 - interx1)

    boxaarea = (y2a - y1a) * (x2a - x1a)
    boxbarea = (y2b - y1b) * (x2b - x1b)

    iou_value = interarea / float(boxaarea + boxbarea - interarea + 1e-10)
    return iou_value


def find_best_match(cname_b, box_b, preds_after, matched):
    """
    Find the best match for a given object (cname_b, box_b) in the preds_after list.
    Returns the best IoU value and index if a suitable match is found.
    """
    best_iou = 0
    best_idx = -1
    for i, (cname_a, _, box_a) in enumerate(preds_after):
        if cname_a == cname_b and not matched[i]:
            current_iou = iou(box_b, box_a)
            if current_iou > best_iou:
                best_iou = current_iou
                best_idx = i
    return best_iou, best_idx


def evaluate_single_image(preds_before, preds_after, iou_threshold=0.5):
    """
    Evaluate the accuracy for a single image by comparing preds_before (reference)
    with preds_after (attacked results).
    """
    matched = [False] * len(preds_after)
    total = len(preds_before)
    correct_count = 0

    for cname_b, _, box_b in preds_before:
        best_iou_val, best_idx = find_best_match(cname_b, box_b, preds_after, matched)
        if best_iou_val >= iou_threshold and best_idx >= 0:
            matched[best_idx] = True
            correct_count += 1

    accuracy = correct_count / total if total > 0 else 0.0
    return accuracy, total, correct_count


def evaluation_per_image(labels_before, labels_after, iou_threshold=0.5):
    """
    Evaluate detection performance on each image individually.
    Uses the detections before perturbation as a pseudo-ground-truth reference.
    
    Returns:
    - results: list of tuples (fname, accuracy, total_reference, correct_matches) per image
    """
    dict_after = {f: preds for f, preds in labels_after}
    results = []

    for fname, preds_before in labels_before:
        preds_after = dict_after.get(fname, [])
        accuracy, total, correct = evaluate_single_image(preds_before, preds_after, iou_threshold)
        results.append((fname, accuracy, total, correct))

    return results


###################################### Main #######################################
labels1 = detection_originale(model, class_names, images, detection_avant_dir)

# labels2 = detection_avec_bruit(model, class_names, images, detection_apres_dir, var=0.01)
labels2 = detection_avec_bruit(model, class_names, images, var=0.01)
# Analyse numérique des résultats

results = evaluation_per_image(labels1, labels2, iou_threshold=0.5)

# Affichage des résultats image par image
for fname, acc, total, correct in results:
    print(f"Image: {fname} | Total objets (ref): {total} | Correctement retrouvés: {correct} | Accuracy: {acc:.2f}")

# On peut également calculer une moyenne sur toutes les images
if len(results) > 0:
    mean_acc = np.mean([r[1] for r in results])
    print(f"\nAccuracy moyenne sur {len(results)} images: {mean_acc*100:.2f}%")

