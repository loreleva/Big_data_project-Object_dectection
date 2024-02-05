import cv2
import numpy as np
import albumentations as A
import time
import yolo3

UPLOAD_FOLDER = 'uploads/'


def generate_frames(filename):
	global yolo_model
	cap = cv2.VideoCapture(UPLOAD_FOLDER+filename)
	fps = cap.get(cv2.CAP_PROP_FPS)
	
	start_time = time.time()
	
	count = 0
	while True:
		is_read, frame = cap.read()
		if not is_read:
			break
		
		img_h, img_w = frame.shape[:-1]
	

		frame = yolo3.get_image_with_bboxes(frame)

		resize_original = A.Compose(
    		[
         		A.Resize(img_h, img_w),
    		],
    		#bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[],),
		)

		frame = resize_original(image=frame)["image"]#, bboxes=bboxes)["image"]
		
		# count fps
		new_time = time.time()
		fps = 1/(new_time - start_time)
		start_time = new_time
		text = f"FPS: {fps:.2f}"

		fontScale = 0.8
		thickness = 1
		font = cv2.FONT_HERSHEY_COMPLEX

		text_size, _ = cv2.getTextSize(text, font, fontScale=fontScale, thickness=thickness)
		text_w, text_h = text_size
		frame = cv2.rectangle(frame, 
							 (0, 0), 
							 (text_w, text_h), 
							 (0, 0, 0), 
							 -1)
		frame = cv2.putText(frame, 
							text, 
							(0, text_h), 
							font, 
							fontScale=fontScale, 
							color=(255, 255, 255),
							thickness=thickness)
		
		ret, buffer = cv2.imencode(".jpg", frame)
		frame = buffer.tobytes()
		yield(b'--frame\r\n'
				b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
