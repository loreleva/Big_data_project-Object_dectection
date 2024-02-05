import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
import os
import pandas as pd
import numpy as np
import albumentations as A
from PIL import Image
import cv2, math
import utils


# resize to give it in input to yolo
image_size = 416
resize = A.Compose(
    [
        A.Resize(image_size, image_size),
        #A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=1),
    ],
    #bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[],),
)

device = "cuda" if torch.cuda.is_available() else "cpu"


yolo_architecture = [
    (32, 3, 1),
    (64, 3, 2),
    ["residual", 1],
    (128, 3, 2),
    ["residual", 2],
    (256, 3, 2),
    ["residualYolo", 8],
    # first yolo route
    (512, 3, 2),
    ["residualYolo", 8],
    # second yolo route
    (1024, 3, 2),
    ["residual", 4],
    # third yolo route
    ["yolo", 1024],
    ["yolo", 512],
    ["yolo", 256],
]


# Convolutional Layer
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bn_act=True, **kwargs):
        super().__init__()
        padding=1 if kernel_size == 3 else 0
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=kernel_size,
                              bias=not bn_act,
                              padding=padding, 
                              **kwargs)
        # if batchnorm, then leaky relu is the activation function
        self.use_bn_act = bn_act
        if self.use_bn_act:
            self.bn = nn.BatchNorm2d(out_channels)
            self.leaky = nn.LeakyReLU(0.1)

    def forward(self, x):
        if self.use_bn_act:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)



# Block of convolutional layers
class ConvBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    ConvLayer(channels, channels // 2, kernel_size=1),
                    ConvLayer(channels // 2, channels, kernel_size=3),
                )
            ]

        self.use_residual = use_residual

    def forward(self, x):
        for layer in self.layers:
            x = layer(x) + self.use_residual * x

        return x





# Block of convolutional layers with feature map to be saved for detections
class ConvBlockYolo(nn.Module):
    def __init__(self, channels, num_repeats=1):
        super().__init__()
        self.layers = nn.ModuleList()
        self.feat_map = None
        num_layer = 1
        for i in range(num_repeats):
            if num_layer == 5:
                self.layers += [
                    ConvLayer(channels, channels // 2, kernel_size=1),
                    ConvLayer(channels // 2, channels, kernel_size=3)
                ]
            else:
                self.layers += [
                    nn.Sequential(
                        ConvLayer(channels, channels // 2, kernel_size=1),
                        ConvLayer(channels // 2, channels, kernel_size=3),
                    )
                ]
            num_layer += 2
            
    def forward(self, x):
        self.feat_map = None
        for layer in self.layers:
            if isinstance(layer, ConvLayer):
                if self.feat_map == None:
                    self.feat_map = layer(x)
                else:
                    x = layer(self.feat_map) + x
            else:
                x = layer(x) + x

        return x



# detection block which outputs bboxes for each grid cell
class DetectionBlock(nn.Module):
    def __init__(self, channels, num_classes):
        super().__init__()
        # feature map to be passed to the the next yolo prediction block
        self.feat_map = None
        # conv layers
        self.convBlock = ConvBlock(channels, use_residual=False, num_repeats=2)
        # penultimate conv block, its feature map must be saved to be passed to other detections block
        self.penultConvBlock = ConvLayer(channels, channels // 2, kernel_size=1)
        self.ultConvBlock = ConvLayer(channels // 2, channels, kernel_size=3)
        # output of last convolution is a tensor: 
        # batch_size x 
        # predictions_size (= n_anchor_boxes * (bbox_coord + obj + n_classes) ) x
        # grid_size x grid_size (13 small objects, 26 medium objects, 52 big objects)
        self.outputPredictions = ConvLayer(channels, 3 * (4+1+num_classes), kernel_size=1, bn_act=False)
    
    def forward(self, x):
        x = self.convBlock(x)
        self.feat_map = self.penultConvBlock(x)
        x = self.ultConvBlock(self.feat_map)
        x = self.outputPredictions(x)
        # change order of dimensions in: batch_size x grid_size x grid_size x predictions_size
        x = torch.permute(x, (0, 2, 3, 1))
        x_shape = x.shape
        # reshape output to: batch_size x grid_size x grid_size x num_anchor_boxes_per_cell x (bbox_coord + obj + n_classes)  
        return x.reshape(x_shape[0], x_shape[1], x_shape[2], 3,  -1)
        



class yolov3(nn.Module):
	def __init__(self, num_classes=10, in_channels=3):
		global yolo_architecture
		super().__init__()
		self.num_classes = num_classes
		self.in_channels = in_channels
		self.layers = self._create_layers(yolo_architecture)
        
	def forward(self, x):
		num_detection = 0
		# list of feature maps values needed for yolo detection
		feat_maps = []
		# tensor of the detections, ordered from the biggest detections to the smallest
		detections = []
		for layer in self.layers:
			x = layer(x)
			# if conv block of darknet, save feature maps used later for detection
			if isinstance(layer, ConvBlockYolo):
				feat_maps.append(layer.feat_map)
                
            # if detection block, save detection and set x to value needed for the next detection
			elif isinstance(layer, DetectionBlock):
				# x is a tensor of dim: n_anchor_boxes * (bbox_coord + obj + n_classes)
				detections.append(x)
				x = layer.feat_map
				num_detection += 1
            # concat darknet feat map with upsampled tensor of previous detection
			elif isinstance(layer, nn.Upsample):
				# feat maps from darkent are ordered from smallest to biggest
				# yolo detection uses feat maps from biggest to smallest
				# then we take darknet feat maps backwards
				x = torch.cat((feat_maps[-num_detection], x), dim=1)
        # return list of detections from the biggest to the smallest
        # each detection is a tensor
		return detections
        
	def _create_layers(self, architecture):
		layers = nn.ModuleList()
		in_channels = self.in_channels
		num_detection = 1
		for module in architecture:
			# conv layer
			if isinstance(module, tuple):
				if module[-1] == "linear":
					bn_act = False
				else:
					bn_act = True
				layers.append(ConvLayer(in_channels, 
										module[0], 
										kernel_size=module[1],
										bn_act=bn_act,
										stride=module[2])
                             )
				in_channels = module[0]
				continue
            
			if isinstance(module, list):
                # residual block
				if module[0] == "residual":
					layers.append(ConvBlock(in_channels,
                                            num_repeats=module[1])
                                 )
					continue
                # residual block with feature map to be saved for detection
				elif module[0] == "residualYolo":
					layers.append(ConvBlockYolo(in_channels,
                                            num_repeats=module[1])
                                 )
					continue
                # detection Block
				elif module[0] == "yolo":
					if num_detection == 3:
						layers.append(DetectionBlock(module[1],
                                                     self.num_classes)
                                     )
					else:
						layers += [DetectionBlock(module[1], 
                                                  self.num_classes),
                                   ConvLayer(module[1] // 2, 
                                             module[1] // 4,
                                             kernel_size=1),
                                   nn.Upsample(scale_factor=2)
                                  ]
					num_detection += 1
					continue
                    
		return layers


model = yolov3().to(device)
model.load_state_dict(torch.load("yolo_weights", map_location=torch.device(device)))
model.eval()


def non_max_suppression(detections, thr_detection, thr_suppression):
	S = [13, 26, 52]
	ANCHORS = [
    	[(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
	    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
	    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
	]
	res_detections = []
	# apply sigmoid to objectness score of each detection
	for s in range(len(S)):
		S_size = S[s]
		# take dimensions of the 3 anchor bboxes corresponding to the grid size
		anchors_w_h = ANCHORS[s]
		# take size of each cell normalized (i.e., in range [0,1])
		cell_size = 1/S_size
		# take detections of the corresponding grid size and remove batch dimension
		S_detections = detections[s].squeeze(0)
		for y in range(S_size):
			for x in range(S_size):
				for idx_anchor in range(3):
					obj_score_detection = torch.sigmoid(S_detections[y][x][idx_anchor][4])
					if obj_score_detection < thr_detection:
						continue
					res_detection = []
					detection_values = S_detections[y][x][idx_anchor]
					# set xcenter, ycenter, width, height, objectness score, class
					res_detection = [(utils.get_bx(detection_values[0], x, cell_size) * cell_size).item(),
                                 (utils.get_by(detection_values[1], y, cell_size) * cell_size).item(),
                                 (utils.get_bw(anchors_w_h[idx_anchor][0], detection_values[2])).item(),
                                 (utils.get_bh(anchors_w_h[idx_anchor][1], detection_values[3])).item(),
                                 (obj_score_detection).item(),
                                 (torch.argmax(detection_values[5:])).item()
                                ]
					res_detections.append(res_detection)
                
	# sort detections by objectness score
	res_detections.sort(key = lambda row: row[4], reverse=True)
	non_suppressed_detections = []
	while len(res_detections) != 0:
    	# take best detection and add it to final detections
		best_detection = res_detections[0]
		non_suppressed_detections.append(best_detection)
		# keep only the non-overlapping detections
		new_res_detections = []
		for detection in res_detections:
			if utils.IOU(best_detection, detection) < thr_suppression:
				new_res_detections.append(detection)
		res_detections = new_res_detections
	return non_suppressed_detections

label_to_color = {
    "human" : (102, 0, 0),
    "barrier" : (0, 0, 0),
    "trafficcone" : (0, 97, 243),
    "bicycle" : (0, 102, 0),
    "bus" : (0, 176, 159),
    "car" : (0, 0, 102),
    "motorcycle" : (84, 105, 0),
    "truck" : (99, 104, 81),
    "construction_vehicles" : (0, 76, 153),
    "trailer" : (76, 153, 0),
}

NUIMAGES_LABELS = [
    "human",
    "barrier",
    "trafficcone",
    "bicycle",
    "bus",
    "car",
    "motorcycle",
    "truck",
    "construction_vehicles",
    "trailer",
]

S = [13, 26, 52]
ANCHORS = [
    	[(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
	    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
	    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
	]

import bisect

def obtain_final_img(img, detections, thr_detection, thr_suppression):
	global NUIMAGES_LABELS, label_to_color, S, ANCHORS
	fontScale = 0.4
	thickness = 1
	font = cv2.FONT_HERSHEY_COMPLEX
	img_h, img_w = img.shape[:2]
	confidence = True

	res_detections = []

	# RETRIEVE DETECTED BBOXES (i.e., the ones for which object score is >= thr_detection)
	# apply sigmoid to objectness score of each detection
	for s in range(len(S)):
		S_size = S[s]
		# take dimensions of the 3 anchor bboxes corresponding to the grid size
		anchors_w_h = ANCHORS[s]
		# take size of each cell normalized (i.e., in range [0,1])
		cell_size = 1/S_size
		# take detections of the corresponding grid size and remove batch dimension
		S_detections = detections[s].squeeze(0)
		# apply sigmoid to objectness score
		S_detections[..., 4] = torch.sigmoid(S_detections[..., 4])
		# compute the class
		S_detections[..., 5] = torch.argmax(S_detections[..., 5:], dim=-1)

		# tensor of x indices
		x_tensor = torch.arange(S_size).unsqueeze(1).repeat(1, 3).to(device)
		# compute bx
		S_detections[..., 0] = (torch.sigmoid(S_detections[..., 0])*cell_size + x_tensor) * cell_size

		# tensor of y indices
		y_tensor = torch.arange(S_size).unsqueeze(1).repeat(1, 3).unsqueeze(1).to(device)
		# compute by
		S_detections[..., 1] = (torch.sigmoid(S_detections[..., 1])*cell_size + y_tensor) * cell_size

		# init tensor of anchors width and height
		tensor_wh = torch.Tensor(anchors_w_h).to(device)
		# compute bw
		S_detections[..., 2:4] = torch.exp(S_detections[..., 2:4]) * tensor_wh

		S_detections = S_detections[S_detections[..., 4] >= thr_detection][..., :6]
		S_detections = S_detections[S_detections[..., 5] == 5]
		res_detections += S_detections.tolist()
		

	# RUN NONMAX SUPPRESSION
	while len(res_detections) != 0:
    	# take best detection and add plot its bbox to the image
		best_detection = res_detections[0]
		bbox = utils.yolo_to_pascal(img_h, img_w, [best_detection], confidence=confidence)[0]
		if confidence:
			# add confidence score to text
			label = NUIMAGES_LABELS[int(bbox[5])]
			text =  label + f" {(bbox[4] * 100):.2f}"
			color = label_to_color[label]
			text = text.upper()
		else:
			text = NUIMAGES_LABELS[int(bbox[4])]
			color = label_to_color[text]
			text = text.upper()
		
		img = cv2.rectangle(img,
                           (bbox[0], bbox[1]),
                           (bbox[2], bbox[3]), 
                            color, 
                            2)
		text_size, _ = cv2.getTextSize(text, 
                                           font, 
                                           fontScale=fontScale, 
                                           thickness=thickness)

		text_w, text_h = text_size
		text_x, text_y = bbox[:2]
            
        # check if text goes out of the image
		if text_x + text_w > img_w:
			text_x = img_w - text_w
		if text_y - text_h < 0:
			text_y = 0
		img = cv2.rectangle(img, 
                                (text_x, text_y), 
                                (text_x + text_w, text_y - text_h), 
                                color, 
                                -1)
		img = cv2.putText(img, 
                              text, 
                              (text_x, text_y), 
                              font, 
                              fontScale=fontScale, 
                              color=(255, 255, 255),
                              thickness=thickness)

		# keep only the non-overlapping detections
		new_res_detections = []
		for detection in res_detections[1:]:
			if utils.IOU(best_detection, detection) < thr_suppression:
				new_res_detections.append(detection)
		res_detections = new_res_detections
	return img

def get_image_with_bboxes(img):
	global model, resize
	
	# img in input is a BGR numpy array of shape (416, 416, 3)
	img = resize(image=img)["image"]
	# transform image channels to RGB
	input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	
	input_img = torch.from_numpy(input_img)
	input_img = torch.permute(input_img, (2, 0, 1)).float()/255
	with torch.no_grad():
		predictions = model(input_img.unsqueeze(0).to(device))
	
	new_img = obtain_final_img(img, predictions, thr_detection=0.97, thr_suppression=0.10)

	return new_img
