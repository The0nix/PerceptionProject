import cv2
import pandas as pd

vidcap = cv2.VideoCapture('video.mp4')
frames_data = pd.read_csv("sensorsFrameTimestamps.csv", header = None)
success,image = vidcap.read()
print(frames_data)
frame_name = str(frames_data.iloc[0,0])
print(frame_name)
cv2.imwrite("frames/"+frame_name+".jpg", image)
count = 0
outF = open("frames.txt", "w")

while success: 
  success,image = vidcap.read()
  cv2.imwrite("frames/"+str(frames_data.iloc[count,0])+".jpg", image)     # save frame as JPEG file     
  outF.write(str(frames_data.iloc[count,0])+".jpg")
  outF.write("\n")
  print('Read a new frame: ', success)
  count += 1

outF.close()
