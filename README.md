# smartFence
Alert before happening
import cv2 
import numpy as np
from datetime import datetime
import vlc
import getpass
import time
import sys
import getpass
import os
import cv2.cuda

script_directory = os.path.dirname(os.path.realpath(__file__))
windowsName = "Out Gate-3"
modelConfiguration = os.path.join(script_directory, 'yolov3.cfg')
modelWeight = os.path.join(script_directory, 'yolov3.weights')
cocoNames = os.path.join(script_directory, 'coco.names')
sirenMusic = os.path.join(script_directory, 'Siren.mp3')
# cameraUsername = input("Enter Camera username: ")
# cameraPassword = getpass.getpass(prompt="Enter Camera Password: ")
# Customization of windows size as per user suitability
# window_width, window_height = None, None
# for i in ["window_width", "window_height"]:
#     while True:
#         try:
#             windowsFrame = int(input(f"Enter {i}: "))
#             if 100 <= windowsFrame <= 1500:
#                 if i == "window_width":
#                     window_width = windowsFrame
#                     break
#                 else:
#                     window_height = windowsFrame
#                     break
#             else:
#                 print(f"{i} must be between 100-1500")
#         except ValueError:
#             print("Invalid input, please enter a valid number!")

cameraUrl = f"rtsp://admin:1234@192.168.1.122:225/Streaming/channels/756"

def cameraResize():
    cv2.namedWindow(windowsName, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(windowsName, 800, 600)
instance = vlc.Instance()
media_list = instance.media_list_new()
media_list_player = instance.media_list_player_new()
siren = instance.media_new(sirenMusic)
media_list.add_media(siren)
media_list_player.set_media_list(media_list)
def play_siren(seconds):
    if media_list_player.is_playing() != 1:
        media_list_player.play()
        time.sleep(seconds)


if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    print("CUDA is available for GPU acceleration. Using GPU acceleration...")
    gpu_is_available_to_be_used = True
else:
    print("CUDA is not available for GPU acceleration. Using CPU...")
    gpu_is_available_to_be_used = False

if gpu_is_available_to_be_used:
    cv2.cuda.setDevice(0)  # Set the device index

try:
    capture = cv2.VideoCapture(cameraUrl)
    # capture.set(cv2.CAP_PROP_FPS, 30) # ----------xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx---------------
    if not capture.isOpened():
        raise Exception("Wrong username/password")
except cv2.error as e:
    print(f"OpenCV Error: {e}")
    play_siren(5)
    sys.exit()
except Exception as e:
    print(f"An error occured : {e}")
    play_siren(5)
    sys.exit()
    cv2.destroyAllWindows
classNames = []
with open(cocoNames, 'rt') as file:
    classNames = file.read().rstrip('\n').split('\n')
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeight)
if gpu_is_available_to_be_used:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
else:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
while True:
    cameraResize()
    today = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    success, image = capture.read()
    # print(image.shape)
    # img = cv2.resize(img, (0,0), None, 0.35, 0.35)
    mask = np.zeros(image.shape, dtype=np.uint8)
    roi_corners = np.array([[(2300,1000), (700, 1000), (400,150),(600,120)]], dtype=np.int32)
    channel_count = image.shape[2]
    # ignore_mask_color = (255)*channel_count
    ignore_mask_color = (255, 255, 255)
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)

    blob = cv2.dnn.blobFromImage(masked_image, 1/255, (320,320), (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)

    layerNames = net.getLayerNames()
    outputNames = [layerNames[i-1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames) # List of predictions made by the neural network 
    hT, wT, cT = image.shape
    bbox=[] # It will store coordinates of bounding boxes
    classIds=[] # It will store class Ids of detected objects
    confs=[] # It will store confidence score of the detected objects
    for output in outputs: # Iterates over each prediction
        for det in output: # Iterates ove each detection in the current prediction (output). Here det means output of 
            # neural network for a single frame of image. Within that output, each det represents information about a
            # detected object or region in that image.
            scores = det[5:] # The det array starting from index 5 containing confidence scores for each class.
            classId=np.argmax(scores) # Third line finds the highest score in the scores array.
            confidence=scores[classId] # The highest score assigned to confidence
            if confidence > 0.5: # Checking the confidence threshold
                w,h = int(det[2]*wT), int(det[3]*hT) # Calculating width and height  of the bounding box using 
                # index of 2 and 3 of the det array and multiplying with image width and height respectively.
                x,y = int((det[0]*wT)-w/2), int((det[1]*wT)-h/2) # The center coordinates x and y are also
                # calculated based on the detected objects position.
                bbox.append([x,y,w,h]) # width (w), height (h) and location (x,y) appended to bbox array.
                classIds.append(classId) # class id of the detected object appended to classIds list.
                confs.append(float(confidence))  # Confidence score of the detected object is stored in confs.
    # print(len(bbox))
    indices = cv2.dnn.NMSBoxes(bbox, confs, 0.5, 0.4) # Indices of the bounding boxes that survived non-maximum
    # suppression process. OpenCV's Deep Neural Networks applying Non-Maximum Suppression
    # 0.5 is the threshold for the confidence score (below this threshold will be disregarded) and 0.4 is the NMS
    # threshold. If the intersection-over-union (IoU) between two boxes is above this threshold, the one with the lower
    # confidence score is suppressed.
    for i in indices:
        box = bbox[i]
        x,y,w,h = box[0], box[1], box[2], box[3]    
        label = str(classNames[classIds[i]])
        if label  in 'person':
            cv2.rectangle(masked_image, (x,y-h), (x+w, y+int(h/2)), (255, 0, 255), 2)
            cv2.putText(masked_image, f'{label.upper()} {int(confs[i]*100)}%', 
                        (x, y-h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2)
            # cv2.imwrite('Images/'+today+'.png', image)
            play_siren(1)
    result = cv2.addWeighted(image, 0.5, masked_image, 0.5, 0)
    cv2.imshow(windowsName, result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
          break
capture.release()
cv2.destroyAllWindows()
