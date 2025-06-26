import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
from ultralytics import YOLO
import onnxruntime as ort
import pickle
import matplotlib
import base64
import zlib
import json
import csv
from io import BytesIO
sys.path.append('..')
from segment_anything import sam_model_registry, SamPredictor
from retinaface.pre_trained_models import get_model
from PIL import Image, ImageDraw, ImageFont
from pycocotools import coco
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerConfig
from transformers import Mask2FormerImageProcessor
from torchvision import models, transforms
import os
import time

image = None
predictor = None
model = None
blank_map = None



def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

def load():  
    global predictorSam, modelYolo, modelOpenCV, mask2former_model, mask2former_processor, maskrcnn_model, modelHead, modelRetina
    
    
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"
                
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
                
    predictorSam = SamPredictor(sam)

    matplotlib.use('TkAgg')
    modelYolo = YOLO('yolov8n.pt')
    
    modelOpenCV = cv2.HOGDescriptor()
    modelOpenCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    mask2former_model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-coco-instance")
    mask2former_processor = Mask2FormerImageProcessor.from_pretrained("facebook/mask2former-swin-large-coco-instance")
    mask2former_model.to(device)
    
    maskrcnn_model = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    
    session_options = ort.SessionOptions()
    session_options.log_severity_level = 3 
    modelHead = YOLO('best.pt')
    
    modelRetina = get_model("resnet50_2020-07-20", max_size=2048)
    modelRetina.eval()
    

def load_image(path):
    global image
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def process_image(image_path, sensitivity, predictorModel, segmentationModel, bbox = None):
    print("predictorModel", predictorModel)
    print("sensitivity", sensitivity)
    model = None
    boxes = None
    cls = None
    conf = None
    original_image = load_image(image_path)
    print("image path:", image_path)
    result_image = original_image.copy()
    all_masks = []
    all_boxes = []
    combined_mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
    if predictorModel == "YOLOv8":
        model = modelYolo
        objects = model(original_image, save=True, classes=[0], device='cpu')
        for result in objects:
            boxes = result.boxes
            cls = boxes.cls
            conf = boxes.conf
            
            for i in range(len(cls)):
                if cls[i] == 0:
                    confidence = conf[i].cpu().numpy()
                    print("confidence", confidence)
                    if confidence < sensitivity:
                        continue
                    input_box = boxes.xyxy[i].cpu().numpy()
                    all_boxes.append(input_box)
                    print("input_box", input_box)
                    
                    mask = get_mask(original_image.copy(), segmentationModel, input_box)
                    if mask is not None:
                        all_masks.append(mask)
    elif predictorModel == "OpenCV":
        model = modelOpenCV
        (objects, weights) = model.detectMultiScale(original_image, winStride=(4, 4), padding=(8, 8), scale=1.05)
        
        for (x, y, w, h), confidence in zip(objects, weights):
            print("confidence", confidence)
            if confidence < sensitivity:
                continue
            input_box = np.array([x, y, x+w, y+h])
            print("input_box", input_box)
            all_boxes.append(input_box)
            mask = get_mask(original_image.copy(), segmentationModel, input_box)
            if mask is not None:
                all_masks.append(mask)
    elif predictorModel == "MaskRCNN":
        model = maskrcnn_model
        model.eval()
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        image_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            predictions = model(image_tensor)
        boxes = predictions[0]['boxes']
        masks = predictions[0]['masks']
        labels = predictions[0]['labels']
        scores = predictions[0]['scores']
        
        people_boxes = boxes[labels == 1]
        people_masks = masks[labels == 1]
        people_scores = scores[labels == 1]
        filtered_boxes = people_boxes[people_scores > sensitivity]
        filtered_masks = people_masks[people_scores > sensitivity]
        for i in range(len(filtered_boxes)):
                input_box = filtered_boxes[i].cpu().numpy()
                x1, y1, x2, y2 = input_box
                
                h, w = original_image.shape[:2]
                x1 = max(0, int(x1) - 10)
                y1 = max(0, int(y1) - 10)
                x2 = min(w, int(x2) + 10)
                y2 = min(h, int(y2) + 10)
                
                all_boxes.append(input_box)
                if(segmentationModel == predictorModel):
                    for mask in filtered_masks:
                        mask_np = mask.squeeze(0).cpu().numpy()
                        boolean_mask = mask_np > 0.5
                        all_masks.append(boolean_mask)
                else:
                        mask = get_mask(original_image.copy(), segmentationModel, input_box)
                        if mask is not None:
                            all_masks.append(mask)
    elif predictorModel == "Mask2Former":
        inputs = mask2former_processor(images=original_image, return_tensors="pt")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = {k: v.to(device) for k, v in inputs.items()}
        mask2former_model.to(device)
        
        with torch.no_grad():
            outputs = mask2former_model(**inputs)
        
        class_queries = outputs.class_queries_logits
        masks_queries = outputs.masks_queries_logits
        
        mask_pred = torch.sigmoid(masks_queries) > 0.5
        
        class_pred = class_queries.softmax(dim=-1)
        
        person_class_id = 0
        
        person_confidence = class_pred[0, :, person_class_id]
        
        for i in range(len(person_confidence)):
                top_class = torch.argmax(class_pred[0, i])
                top_score = class_pred[0, i, top_class]

                if top_class == person_class_id and top_score >= sensitivity:
                    instance_mask = mask_pred[0, i].cpu().numpy()
                    
                    print(f"Mask shape: {instance_mask.shape}, Positive pixels: {np.sum(instance_mask)}")
                    
                    if instance_mask.shape != original_image.shape[:2]:
                        instance_mask = cv2.resize(
                            instance_mask.astype(np.uint8),
                            (original_image.shape[1], original_image.shape[0]),
                            interpolation=cv2.INTER_NEAREST
                        )
                    
                    if instance_mask.sum() > 100:
                        y_indices, x_indices = np.where(instance_mask)
                        if len(y_indices) > 0 and len(x_indices) > 0:
                            x1, y1 = np.min(x_indices), np.min(y_indices)
                            x2, y2 = np.max(x_indices), np.max(y_indices)
                            
                            h, w = original_image.shape[:2]
                            x1 = max(0, min(w-1, x1))
                            y1 = max(0, min(h-1, y1))
                            x2 = max(0, min(w-1, x2))
                            y2 = max(0, min(h-1, y2))
                            
                            pad = 10
                            x1 = max(0, x1 - pad)
                            y1 = max(0, y1 - pad)
                            x2 = min(w, x2 + pad)
                            y2 = min(h, y2 + pad)
                            
                            input_box = np.array([x1, y1, x2, y2])
                            all_boxes.append(input_box)
        if segmentationModel == "Mask2Former":
            for i in range(len(person_confidence)):
                if person_confidence[i] >= sensitivity:
                    instance_mask = mask_pred[0, i].cpu().numpy()
                    
                    if instance_mask.shape != original_image.shape[:2]:
                        instance_mask = cv2.resize(
                            instance_mask.astype(np.uint8),
                            (original_image.shape[1], original_image.shape[0]),
                            interpolation=cv2.INTER_NEAREST
                        )
                        
                    if instance_mask.sum() > 0:
                        all_masks.append(instance_mask)
        else:
                            print(f"Bounding box for mask {i}: {input_box}")
                            
                            mask = get_mask(original_image.copy(), segmentationModel, input_box)
                            if mask is not None:
                                all_masks.append(mask)
    elif predictorModel == "Head":
        model = modelHead
        objects = model(original_image, save=True, classes=[0], device='cpu')
        for result in objects:
            boxes = result.boxes
            cls = boxes.cls
            conf = boxes.conf
            
            for i in range(len(cls)):
                if cls[i] == 0:
                    confidence = conf[i].cpu().numpy()
                    print("confidence", confidence)
                    if confidence < sensitivity:
                        continue
                    input_box = boxes.xyxy[i].cpu().numpy()
                    
                    input_box[0] -= 4
                    input_box[1] -= 4
                    input_box[2] += 4
                    input_box[3] += 4
                    print("input_box", input_box)
                    all_boxes.append(input_box)
                    
                    mask = get_mask(original_image.copy(), segmentationModel, input_box)
                    if mask is not None:
                        all_masks.append(mask)
    elif predictorModel == "Manual" and bbox is not None:
        all_boxes.append(bbox)
        mask = get_mask(original_image.copy(), segmentationModel, bbox)
        if mask is not None:
            all_masks.append(mask)
    elif predictorModel == "Retina":
        model = modelRetina
        with torch.no_grad():
            detections = model.predict_jsons(image)

        for detection in detections:
            bbox = detection["bbox"]
            if len(bbox) != 4:
                continue

            x1, y1, x2, y2 = map(int, bbox)
            h, w = original_image.shape[:2]
            x1 = max(0, x1 - 10)
            y1 = max(0, y1 - 10)
            x2 = min(w, x2 + 10)
            y2 = min(h, y2 + 10)

            input_box = np.array([x1, y1, x2, y2])
            all_boxes.append(input_box)
            mask = get_mask(original_image.copy(), segmentationModel, input_box)
            if mask is not None:
                all_masks.append(mask)
    else:
        print("Invalid predictor model or no bounding box provided.")
    if len(all_masks) > 0:
        for i in range(len(all_masks)):
            mask_shape = all_masks[i].shape
            img_shape = original_image.shape[:2]
            
            if mask_shape != img_shape:
                print(f"Resizing mask from {mask_shape} to {img_shape}")
                all_masks[i] = cv2.resize(
                    all_masks[i].astype(np.uint8),
                    (img_shape[1], img_shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )
    if len(all_masks) > 0:
        for mask in all_masks:
            if mask.ndim > 2:
                mask = mask.squeeze()
            
            mask = mask.astype(np.uint8)
            mask = mask > 0
            
            mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
            purple_overlay = np.array([255, 0, 220], dtype=np.uint8)
            result_image = np.where(mask_3d, purple_overlay, result_image)
            
            combined_mask = cv2.bitwise_or(combined_mask, mask.astype(np.uint8))
            
        return Image.fromarray(result_image), combined_mask, all_boxes
    return Image.fromarray(original_image), combined_mask, all_boxes

def get_mask(image, segmentationModel, input_box):
    print("segmentationModel", segmentationModel)
    segmented_image = np.array(image)
    
    if segmentationModel == "SAM":
        predictorSam.set_image(segmented_image)
        
        masks, _, _ = predictorSam.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )
        return masks[0]
    elif segmentationModel == "OpenCV":
        mask = np.zeros(segmented_image.shape[:2], np.uint8)
        rect = (int(input_box[0]), int(input_box[1]), int(input_box[2]-input_box[0]), int(input_box[3]-input_box[1]))
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)
        cv2.grabCut(segmented_image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        return mask2
    else:
        print("Invalid segmentation model")
    return None

def write_results(image, detectionModel, segmentationModel, loadingTime, dice, iou, tp, fp, tn, fn, max_memory, height, width):
    imagename = os.path.splitext(os.path.basename(image))[0]
    filename = f"{detectionModel}_{segmentationModel}.csv"
    try:
        with open(filename, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([imagename, loadingTime, dice, iou, tp, fp, tn, fn, max_memory / 10**6, height, width])
    except Exception as e:
        print(f"Error writing to file {filename}: {e}")
    
def clear_results_file(detectionModel, segmentationModel):
    filename = f"{detectionModel}_{segmentationModel}.csv"
    try:
        with open(filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["image_name", "loading_time", "dice", "iou", "tp", "fp", "tn", "fn", "max_memory (MB)", "height", "width"])
    except Exception as e:
        print(f"Error clearing file {filename}: {e}")
def compare_masks(new_mask, original_mask, new_box, original_box, bmp, type):
    if type == "segmentation":
        if(bmp == False):
            if original_mask.max() > 1:
                original_mask = (original_mask > 0).astype(np.uint8)
            
        intersection = np.sum(np.logical_and(original_mask, new_mask))
        total = np.sum(original_mask) + np.sum(new_mask)
        union = np.sum(np.logical_or(original_mask, new_mask))
        tp = np.sum(np.logical_and(original_mask, new_mask))
        tn = np.sum(np.logical_and(original_mask == 0, new_mask == 0))
        fp = np.sum(np.logical_and(original_mask == 0, new_mask == 1))
        fn = np.sum(np.logical_and(original_mask == 1, new_mask == 0))
        dice = 2.0 * intersection / total if total > 0 else 0
        iou = intersection / union if union > 0 else 0
    elif type == "detection":
        intersection = np.sum(np.logical_and(original_box, new_box))
        union = np.sum(np.logical_or(original_box, new_box))
        tp = intersection
        fp = np.sum(new_box) - intersection
        fn = np.sum(original_box) - intersection
        tn = 0
        dice = 2.0 * intersection / (np.sum(original_box) + np.sum(new_box)) if union > 0 else 0
        iou = intersection / union if union > 0 else 0
    return dice, iou, tp, tn, fp, fn
def create_blank_map():
    global blank_map
    if hasattr(image, 'shape'):
        blank_map = np.zeros(image.shape[:2], dtype=np.uint8)
    elif hasattr(image, 'size'):
        width, height = image.size
        blank_map = np.zeros((height, width), dtype=np.uint8)
    else:
        blank_map = np.zeros((500, 500), dtype=np.uint8)
    return blank_map
def decode_bitmap(bitmap_data, xoffset, yoffset):
    global blank_map
    if blank_map is None:
        blank_map = create_blank_map()
        
    decoded_data = base64.b64decode(bitmap_data)
    decompressed_data = zlib.decompress(decoded_data)
    binary_mask = Image.open(BytesIO(decompressed_data)).convert('1')
    binary_np = np.array(binary_mask)
    combine_bitmap(binary_np, xoffset, yoffset)
    return blank_map
def create_polygon_mask(points, image_shape):
    poly_mask = np.zeros(image_shape, dtype=np.uint8)
    polygon = np.array(points, dtype=np.int32)
    cv2.fillPoly(poly_mask, [polygon], 1)
    if "inaterior" in points:
        for interior in points["inaterior"]:
            interior_polygon = np.array(interior, dtype=np.int32)
            cv2.fillPoly(poly_mask, [interior_polygon], 0)
    return poly_mask 
def reset_blank_map():
    """Reset the blank map to prepare for a new image or annotation file"""
    global blank_map
    blank_map = None
    return create_blank_map()
def get_bitmapdata_supervisely(filejson):
    """Extract bitmap data from all objects in a Supervisely JSON file"""
    with open(filejson, 'r') as file:
        data = json.load(file)
    
    reset_blank_map()
    
    for obj in data["objects"]:
        geomtype = obj["geometryType"]
        if geomtype in obj and "data" in obj[geomtype]:
            info = obj[geomtype]["data"]
            xoffset = obj[geomtype]["origin"][0]
            yoffset = obj[geomtype]["origin"][1]
            
            decoded_data = base64.b64decode(info)
            decompressed_data = zlib.decompress(decoded_data)
            binary_mask = Image.open(BytesIO(decompressed_data)).convert('1')
            binary_np = np.array(binary_mask)
            combine_bitmap(binary_np, xoffset, yoffset)
        elif geomtype == "polygon":
            points = obj["points"]["exterior"]
            polygon_mask = create_polygon_mask(points, blank_map.shape)
            combine_bitmap(polygon_mask, 0, 0)    
    
    return get_combined_bitmap()
def combine_bitmap(additional_bitmap, x_offset, y_offset):
    global blank_map
    if blank_map is None:
        blank_map = create_blank_map()
    
    if hasattr(additional_bitmap, 'shape'):
        bitmap_height, bitmap_width = additional_bitmap.shape[:2]
    else:
        bitmap_width, bitmap_height = additional_bitmap.size
    padded_mask = np.zeros_like(blank_map, dtype=np.uint8)
    
    end_y = min(y_offset + bitmap_height, blank_map.shape[0])
    end_x = min(x_offset + bitmap_width, blank_map.shape[1])
    
    valid_height = end_y - y_offset
    valid_width = end_x - x_offset
    
    if valid_height > 0 and valid_width > 0:
        padded_mask[y_offset:end_y, x_offset:end_x] = additional_bitmap[:valid_height, :valid_width]
    
    blank_map = np.logical_or(blank_map, padded_mask).astype(np.uint8)
def get_combined_bitmap():
    return blank_map
def convert_box_to_bitmapdata(box, image_shape):
    
    x1, y1, x2, y2 = box
    mask = np.zeros(image_shape, dtype=np.uint8)
    cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), 1, -1)
    return mask
def combine_box_bitmapdata(bitmapdata):
    blank_map = create_blank_map()
    
    for box in bitmapdata:
        if len(box) == 4:
            x1, y1, x2, y2 = map(int, box)
            blank_map[y1:y2, x1:x2] = 1
    return blank_map
def load_boxes_from_path(path):
    boxes = []
    create_blank_map()

    try:
        with open(path, 'r') as file:
            data = json.load(file)
            
        for entry in data:
            if len(entry) != 4:
                print(f"Warning: Skipping invalid entry: {entry}")
                continue
            x1, y1, x2, y2 = map(int, entry)
            bbox = np.array([x1, y1, x2, y2])
            boxes.append(bbox)
    except Exception as e:
        print(f"Error loading JSON file {path}: {e}")
    
    return boxes
    
    
        
def save_results(image, mask, iou, dice, segmentation_alg, detection_alg):
    imagename = os.path.splitext(os.path.basename(image))[0]
    color = np.array([30/255, 144/255, 255/255, 0.6])
    image = load_image(image)
    image = np.array(image)
    h, w = mask.shape[-2:]
    mask_image=mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    output_image = np.copy(image)
    output_image = output_image.astype(np.float32)/255.0
    mask_overlayed_image = output_image * (1 - mask_image[:, :, 3:]) + mask_image[:, :, :-1] * mask_image[:, :, :-1]
    mask_overlayed_image = (mask_overlayed_image * 255).astype(np.uint8)
    mask_overlayed_image = Image.fromarray(mask_overlayed_image)
    image = Image.fromarray(image)
    combined_image = Image.new("RGB", (2*mask_overlayed_image.width, mask_overlayed_image.height))
    combined_image.paste(image, (0,  0))
    combined_image.paste(mask_overlayed_image, (image.width, 0))
    draw = ImageDraw.Draw(combined_image)
    font = ImageFont.truetype("arial.ttf", size=24)
    text = f"IoU: {iou:.4f}\nDice: {dice:.4f}"
    outline_color = "black"
    text_position = (10, mask_overlayed_image.height - 50)
    for offset in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        draw.text((text_position[0] + offset[0], text_position[1] + offset[1]), text, font=font, fill=outline_color)
    text_color = "white"
    draw.text(text_position, text, font=font, fill=text_color)
    filepath = f"{segmentation_alg}_{detection_alg}"
    filename = f"{imagename}.jpg"
    fullpath = os.path.join(filepath, filename)
    os.makedirs(filepath, exist_ok=True)
    combined_image.save(fullpath)
    