import numpy as np
import math
from ..utils.box import BoundBox

def expit(x):
    return 1. / (1. + np.exp(-x))

def box_intersection(a, b):
    w = max(0, min(a[0] + a[2]/2, b[0] + b[2]/2) - max(a[0] - a[2]/2, b[0] - b[2]/2))
    h = max(0, min(a[1] + a[3]/2, b[1] + b[3]/2) - max(a[1] - a[3]/2, b[1] - b[3]/2))
    return w * h

def box_union(a, b):
    i = box_intersection(a, b)
    return a[2] * a[3] + b[2] * b[3] - i

def box_iou(a, b):
    return box_intersection(a, b) / (box_union(a, b) + 1e-6)

def nms(final_probs, final_bbox, iou_threshold=0.4):
    """
    Vectorized NumPy implementation of Non-Maximum Suppression.
    """
    class_length = final_probs.shape[1]
    boxes = []
    
    # Coordinates for all boxes
    x = final_bbox[:, 0]
    y = final_bbox[:, 1]
    w = final_bbox[:, 2]
    h = final_bbox[:, 3]
    
    # Calculate areas once
    areas = w * h
    
    # Left, top, right, bottom coordinates
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    
    for c in range(class_length):
        probs = final_probs[:, c]
        order = probs.argsort()[::-1]
        keep = []
        
        while order.size > 0:
            i = order[0]
            if probs[i] == 0: break
            keep.append(i)
            
            if order.size == 1: break
            
            # Intersection coordinates
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            # Intersection width and height
            ww = np.maximum(0.0, xx2 - xx1)
            hh = np.maximum(0.0, yy2 - yy1)
            inter = ww * hh
            
            # IoU
            ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            
            # Filter boxes with high IoU
            inds = np.where(ovr <= iou_threshold)[0]
            order = order[inds + 1]
            
        for i in keep:
            bb = BoundBox(class_length)
            bb.x, bb.y, bb.w, bb.h = x[i], y[i], w[i], h[i]
            bb.c = final_bbox[i, 4]
            bb.probs = np.zeros(class_length)
            bb.probs[c] = probs[i]
            boxes.append(bb)
            
    return boxes

def box_constructor(meta, net_out_in):
    """
    Pure Python implementation of YOLOv2 box constructor.
    """
    H, W, _ = meta['out_size']
    C = meta['classes']
    B = meta['num']
    threshold = meta['thresh']
    anchors = np.asarray(meta['anchors']).reshape(-1, 2)
    
    # Reshape input
    net_out = net_out_in.reshape([H, W, B, -1])
    
    probs = np.zeros((H, W, B, C), dtype=np.float32)
    bboxes = np.zeros((H, W, B, 5), dtype=np.float32)
    
    # Vectorized computation for Bbox
    # tx, ty, tw, th, to
    tx = net_out[..., 0]
    ty = net_out[..., 1]
    tw = net_out[..., 2]
    th = net_out[..., 3]
    to = net_out[..., 4]
    
    conf = expit(to)
    
    # x, y relative to cell
    grid_x = np.tile(np.arange(W), (H, 1))
    grid_y = np.tile(np.arange(H).reshape(-1, 1), (1, W))
    
    bboxes[..., 0] = (grid_x[:, :, np.newaxis] + expit(tx)) / W
    bboxes[..., 1] = (grid_y[:, :, np.newaxis] + expit(ty)) / H
    bboxes[..., 2] = np.exp(tw) * anchors[:, 0] / W
    bboxes[..., 3] = np.exp(th) * anchors[:, 1] / H
    bboxes[..., 4] = conf
    
    # Class probabilities
    classes = net_out[..., 5:]
    # Softmax over classes
    exp_classes = np.exp(classes - np.max(classes, axis=-1, keepdims=True))
    class_probs = exp_classes / np.sum(exp_classes, axis=-1, keepdims=True)
    
    # Multiply by confidence
    final_probs = class_probs * conf[..., np.newaxis]
    
    # Filter by threshold
    mask = final_probs > threshold
    probs[mask] = final_probs[mask]
    
    # NMS
    return nms(probs.reshape(-1, C), bboxes.reshape(-1, 5), iou_threshold=0.4)

def yolo_box_constructor(meta, net_out, threshold):
    """
    Pure Python implementation of YOLO (v1) box constructor.
    """
    sqrt = meta['sqrt'] + 1
    C, B, S = meta['classes'], meta['num'], meta['side']
    SS = S * S
    prob_size = SS * C
    conf_size = SS * B
    
    probs = net_out[:prob_size].reshape([SS, C])
    confs = net_out[prob_size : prob_size + conf_size].reshape([SS, B])
    coords = net_out[prob_size + conf_size :].reshape([SS, B, 4])
    
    final_probs = np.zeros([SS, B, C], dtype=np.float32)
    final_coords = np.zeros([SS, B, 4], dtype=np.float32)
    
    for grid in range(SS):
        for b in range(B):
            grid_x = grid % S
            grid_y = grid // S
            final_coords[grid, b, 0] = (coords[grid, b, 0] + grid_x) / S
            final_coords[grid, b, 1] = (coords[grid, b, 1] + grid_y) / S
            final_coords[grid, b, 2] = coords[grid, b, 2] ** sqrt
            final_coords[grid, b, 3] = coords[grid, b, 3] ** sqrt
            
            p = probs[grid] * confs[grid, b]
            mask = p > threshold
            if np.any(mask):
                final_probs[grid, b, mask] = p[mask]
    
    return nms(final_probs.reshape(-1, C), final_coords.reshape(-1, 4), iou_threshold=0.4)
