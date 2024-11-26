from ultralytics import YOLO
import numpy as np
import cv2
import time
import datetime
from collections import defaultdict
# Save username and password in another file
from login_info import username, password

def decode_fourcc(v):
    v = int(v)
    return "".join([chr((v >> 8 * i) & 0xFF) for i in range(4)])

model = YOLO('downloaded_models/yolo11n.pt') # Load YOLOv11 model
RTSP_URL = 'rtsp://'+username+':'+password+'@192.168.1.201:554/Streaming/Channels/501'
cap = cv2.VideoCapture(RTSP_URL)
start_time2 = time.time()
# Get video properties
fourcc = decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC))
fps = int(cap.get(cv2.CAP_PROP_FPS))
focus = cap.get(cv2.CAP_PROP_FOCUS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("Stream prop: cc {}, fps {}, focus {}, w {}, h{}".format(fourcc, fps, focus, w, h))
fourcc_out = cv2.VideoWriter_fourcc('M','J','P','G') # Define the codec
TIME_NOW = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#out_track_file = 'outputs/'+TIME_NOW+'_track.avi'
#out_track = cv2.VideoWriter(#out_track_file, fourcc_out, fps, (w,h))
num_frames = 0
names = model.names
track_images = defaultdict(lambda: [])
track_classes = defaultdict(lambda: [])

while(cap.isOpened()):
    ret, frame = cap.read()
    start_time = time.time()
    if ret==True:
        num_frames += 1
        results = model.track(source=frame, conf=0.5, persist=True) # Run YOLOv11 inference
        cv2.imshow("YOLOv11 Real-Time Detection", results[0].plot()) # Display detections
        loopTime = time.time() - start_time
        if len(results[0].boxes.data) != 0:
            for result in results:
                if result.boxes.id != None:
                    for id, classi, box in zip(result.boxes.id, result.boxes.cls, result.boxes.xyxy):
                        b = box.numpy().astype(np.uint)
                        print(f"ID: {int(id)}, Class: {names[int(classi)]}, Bounding Box: {b}")
                        crop = frame[b[1]:b[3], b[0]:b[2]]
                        if(crop.size > np.array(track_images[int(id)]).size):
                            track_images[int(id)] = crop
                            track_classes[int(id)] = int(classi)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            if len(track_classes) != 0:
                TIME_NOW_2 = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                for idd in track_classes:
                    c = track_classes[idd]
                    if c != []:
                        im = track_images[idd]
                        filename = 'outputs/images/'+TIME_NOW_2+'_'+str(int(idd))+'_'+names[c]+'.jpg'
                        cv2.imwrite(filename, im)
            break

        # write the tracked frame twice to fix out fps = 1/2 input fps
        #out_track.write(results[0].plot())
        #out_track.write(results[0].plot())
    else:
        break

# Release everything if job is finished
etime=time.time()-start_time2
print("Total elapsed time: ", etime)
print("Total num of frames: ", num_frames)
cap.release()
#out_track.release()
cv2.destroyAllWindows()