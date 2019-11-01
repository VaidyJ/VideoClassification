# USAGE

# python predict_video.py --model model/activity_gpu.model --label-bin model/lb.pickle --input example_clips/cyclone_movie.mp4 --output output/cyclon_movie_output.avi --size 10

# import the necessary packages
from tensorflow.keras.models import load_model
from collections import deque
import numpy as np
import argparse
import pickle
import cv2
from twilio.rest import Client
import csv
import pandas as pd
import numpy as np
import datetime
camid = 'IRNSS-1I'
location = 'Geosynchronous / 55°E, 29° inclined orbit'

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained serialized model")
ap.add_argument("-l", "--label-bin", required=True,
	help="path to  label binarizer")
ap.add_argument("-i", "--input", required=True,
	help="path to our input video")
ap.add_argument("-o", "--output", required=False,
	help="path to our output video")
ap.add_argument("-s", "--size", type=int, default=128,
	help="size of queue for averaging")
args = vars(ap.parse_args())

# load the trained model and label binarizer from disk
print("[INFO] loading model and label binarizer...")
model = load_model(args["model"])
lb = pickle.loads(open(args["label_bin"], "rb").read())

# initialize the image mean for mean subtraction along with the
# predictions queue
mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
Q = deque(maxlen=args["size"])

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)
client = Client("ACea4cecca40ebb1bf4594098d5cef4541", "32789639585561088d5937514694e115") #update from twilio
prelabel = ''
ok = 'Normal'
fi_label = []
framecount = 0

frames = []
shots = []
player = []

playerNo = round(np.random.rand()*100)

forehandCount = 0
backhandCount = 0

# loop over frames from the video file stream


while True:
	# read the next frame from the file
	(grabbed, frame) = vs.read()

	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break

	# if the frame dimensions are empty, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]
	framecount = framecount+1
	# clone the output frame, then convert it from BGR to RGB
	# ordering, resize the frame to a fixed 224x224, and then
	# perform mean subtraction
	output = frame.copy()
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	frame = cv2.resize(frame, (224, 224)).astype("float32")
	frame -= mean

	# make predictions on the frame and then update the predictions
	# queue
	preds = model.predict(np.expand_dims(frame, axis=0))[0]
#	print('Preds = :', preds)
	
#	total = (preds[0]+ preds[1]+preds[2] + preds[3]+ preds[4]+preds[5])
#	maximum = max(preds)
#	rest = total - maximum
    
#	diff = (.8*maximum) - (.1*rest)
#	print('Difference of prob ', diff)
#	th = 100
#	if diff > .60:
#		th = diff
#	print('Old threshold = ', th)
    
    
	prediction = preds.argmax(axis=0)
	Q.append(preds)

	# perform prediction averaging over the current history of
	# previous predictions
	results = np.array(Q).mean(axis=0)
	print('Results = ', results)
	maxprob = np.max(results)
	print('Maximun Probability = ', maxprob)
	i = np.argmax(results)
	print("i = ", i)
	label = lb[i]

	rest = 1 - maxprob
    
	diff = (maxprob) - (rest)
	#print('Difference of prob ', diff)
	th = 100
	if diff > .80:
		th = diff
      
	frames.append(framecount)
	player.append(playerNo)
	shots.append(i)
        
        
	print(label, ':' , prelabel)   
	if framecount > 10:
		if label != prelabel:
			if (i==0):
				backhandCount += 1
			else:
				forehandCount += 1
    
	textCategory = "{} - {:.2f}%".format((label), maxprob * 100)
	#text = ""
	textBackhandCount = "Backhand count : " + str(backhandCount)
	textForehandCount = "Forehand count : " + str(forehandCount)
	cv2.putText(output, textCategory, (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 5)
	cv2.putText(output, textForehandCount, (35, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 5)
	cv2.putText(output, textBackhandCount, (35, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 5)
	


	prelabel = label    



	# check if the video writer is None
	if writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		#writer = cv2.VideoWriter(args["output"] + datetime.datetime.now().strftime('%b-%d-%Y_%H%M'), fourcc, 30, (W, H), True)
		writer = cv2.VideoWriter( "output/"+ args["input"].split('/')[1].split('.')[0] + "_" + args["model"]  + "_output" + '.MOV', fourcc, 30, (W, H), True)
		
		

	# write the output frame to disk
	writer.write(output)

	# show the output image
	cv2.imshow("Output", output)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
print('Frame count', framecount)
#print('Count label', fi_label)
# release the file pointers
print("[INFO] cleaning up...")
#print( "output/"+ args["input"].split('/')[1].split('.')[0] + "_" + args["model"]  + "_output" + '.MOV')
writer.release()
vs.release()

df = pd.DataFrame(columns =[ 'Player' ,'FrameNo', 'Shot'])

df['Player'] = player
df['Shot'] = shots
df['FrameNo'] = frames

outFileName = str(playerNo) + ".csv"

df.to_csv(outFileName)