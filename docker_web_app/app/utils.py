import numpy as np

def normalize_bboxes(img_h, img_w, bboxes, confidence=False):
    # transform elements of bboxes in range [0,1]
    new_bboxes = []
    for bbox in bboxes:
        if confidence:
            new_bboxes.append([bbox[0]/img_w, bbox[1]/img_h, bbox[2]/img_w, bbox[3]/img_h, bbox[4], bbox[5]])
        else:
            new_bboxes.append([bbox[0]/img_w, bbox[1]/img_h, bbox[2]/img_w, bbox[3]/img_h, bbox[4]])
    return new_bboxes

def unnormalize_bboxes(img_h, img_w, bboxes, confidence=False):
    # transform elements of bboxes from range [0,1] in range of integers [0, img_w], [0, img_h]
    new_bboxes = []
    for bbox in bboxes:
        if confidence:
            new_bboxes.append([int(bbox[0]*img_w), int(bbox[1]*img_h), int(bbox[2]*img_w), int(bbox[3]*img_h), bbox[4], bbox[5]])
        else:
            new_bboxes.append([int(bbox[0]*img_w), int(bbox[1]*img_h), int(bbox[2]*img_w), int(bbox[3]*img_h), bbox[4]])
    return new_bboxes
    
def pascal_to_yolo(img_h, img_w, bboxes, confidence=False):
    # convert bboxes from pascal notation to yolo notation
    # i.e., from [xmin, ymin, xmax, ymax, category] notation to [xcenter, ycenter, width, height, category] normalized
    new_bboxes = []
    for bbox in bboxes:
        # check if bbox exceed image bounds
        if bbox[2] > img_w:
            bbox[2] = img_w
        if bbox[3] > img_h:
            bbox[3] = img_h
        bbox_h = bbox[3] - bbox[1]
        bbox_w = bbox[2] - bbox[0]
        if bbox_h == 0 or bbox_w == 0:
            continue
        if confidence:
            new_bboxes.append([bbox[0] + bbox_w/2, 
                           bbox[1] + bbox_h/2, 
                           bbox_w,
                           bbox_h, 
                           bbox[4], 
                           bbox[5]
                          ])
        else:
            new_bboxes.append([bbox[0] + bbox_w/2, 
                           bbox[1] + bbox_h/2, 
                           bbox_w,
                           bbox_h, 
                           bbox[4]
                          ])
    return normalize_bboxes(img_h, img_w, new_bboxes, confidence=confidence)
    
def yolo_to_pascal(img_h, img_w, bboxes, confidence=False):
    # convert bboxes from yolo notation to pascal notation
    # i.e., from [xcenter, ycenter, width, height, category] notation to [xmin, ymin, xmax, ymax, category] notation
    new_bboxes = []
    for bbox in bboxes:
        if bbox[2] == 0 or bbox[3] == 0:
            continue
        if confidence:
            new_bboxes.append([bbox[0] - bbox[2]/2, 
                           bbox[1] - bbox[3]/2, 
                           bbox[0] + bbox[2]/2,
                           bbox[1] + bbox[3]/2, 
                           bbox[4],
                           bbox[5]
                          ])
        else:
            new_bboxes.append([bbox[0] - bbox[2]/2, 
                           bbox[1] - bbox[3]/2, 
                           bbox[0] + bbox[2]/2,
                           bbox[1] + bbox[3]/2, 
                           bbox[4]
                          ])
    return unnormalize_bboxes(img_h, img_w, new_bboxes, confidence=confidence)


def IOU(bbox1, bbox2, img_h=416, img_w=416, confidence=True):
    # transform boxes in [xmin, ymin, xmax, ymax]
    bbox1 = yolo_to_pascal(img_h, img_w, [bbox1], confidence=confidence)[0]
    bbox2 = yolo_to_pascal(img_h, img_w, [bbox2], confidence=confidence)[0]
    
    xA = max(bbox1[0], bbox2[0])
    yA = max(bbox1[1], bbox2[1])
    xB = min(bbox1[2], bbox2[2])
    yB = min(bbox1[3], bbox2[3])
    
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    
    box1Area = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
    box2Area = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)
    
    iou = interArea / float(box1Area + box2Area - interArea)
    
    return iou



def sigmoid(x):
    return 1 / (1 + 1 / np.exp(x))

def inverse_sigmoid(x):
    return np.log(x/(1-x))

# functions needed to obtain bbox coordinates from YOLO's output
# tx, ty, tw, th are the YOLO's output values, needed to compute bx, by, bw, bh which are the bbox coordinates

def get_tx(b_x, c_x):
    return inverse_sigmoid(b_x - c_x)

def get_ty(b_y, c_y):
    return inverse_sigmoid(b_y - c_y)

def get_bx(t_x, c_x, cell_size):
    return sigmoid(t_x)*cell_size + c_x

def get_by(t_y, c_y, cell_size):
    return sigmoid(t_y)*cell_size + c_y

def get_bw(p_w, t_w):
    return p_w * np.exp(t_w)

def get_bh(p_h, t_h):
    return p_h * np.exp(t_h)

def get_tw(b_w, p_w):
    return np.log(b_w/p_w)

def get_th(b_h, p_h):
    return np.log(b_h/p_h)


