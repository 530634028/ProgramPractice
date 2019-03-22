
"""
  emotion detection in real-time
  a: zhong
  date:2019-3-20

"""

# import the necessary packages
import sys
sys.path.append("F:\ProgramPractice\DLFCV")

from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True,
                help="path to where the face cascade resides")
ap.add_argument("-m", "--model", required=True,
                help="path to pre-trained emotion detector CNN")
ap.add_argument("-v", "--video",
                help="path to the (optional) video file")
args = vars(ap.parse_args())

# load the face detector cascade, emotion detection CNN, then define
# the list of emotion labels
detector = cv2.CascadeClassifier(args["cascade"])
model = load_model(args["model"])
EMOTIONS = ["angry", "scared", "happy", "sad", "surprised", "neutral"]

# if a video path was not supplied, grab the reference to the webcam
if not args.get("video", False):
    camera = cv2.VideoCapture(1)

# otherwise, load the video
else:
    camera = cv2.VideoCapture(args["video"])  # args["video"]

# keep looping
while True:
    # grab the current frame
    (grabbed, frame) = camera.read()

    ##    cv2.imshow('image', frame)

    # if we are viewing a video and we did not grab a frame, then we
    # have reached the end of the video
    if  args.get("video") and not grabbed: #
       break

    # resize the frame, convert it to grayscale, and then clone the
    # original frame so we can draw on it later in the program
    frame = imutils.resize(frame, width=300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    canvas = np.zeros((200, 300, 3), dtype="uint8")
    frameClone = frame.copy()

    # cv2.imshow("original face data", frameClone)
    # cv2.waitKey(1)

    # detect faces in the input frame, then clone the frame so that
    # we can draw on it
    if detector.empty():
        raise IOError('Unable to load the face cascade classifier xml file')
    rects = detector.detectMultiScale(gray, scaleFactor=1.3,
                                      minNeighbors=2, minSize=(30, 30), # 5
                                      flags=cv2.CASCADE_SCALE_IMAGE)
    # print(len(rects))

    # ensure at last one face was found before continuing
    if len(rects) > 0:
        # determine the largest face area
        rect = sorted(rects, reverse=True,
                      key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]

        # test if there have face detected by cascade
        # print(rect)

        (fX, fY, fW, fH) = rect

        # extract ROI of face, resize it to fixed 28x28 pixels
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # display the roi we extracted
        # cv2.imshow("face roi", roi)

        # make a prediction on the ROI, then lookup the class
        # label
        preds = model.predict(roi)[0]
        label = EMOTIONS[preds.argmax()]

        # loop over the labels + probabilities and draw them
        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
            # construct the label text
            text = "{}: {:.2f}%".format(emotion, prob * 100)

            # draw the label + probabiities bar on the canvas
            w = int(prob * 300)
            cv2.rectangle(canvas, (5, (i * 35) + 5),
                                   (w, (i * 35) + 35), (0, 0, 255), -1)
            cv2.putText(canvas, text, (10, (i * 35) + 23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 2)

            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(frameClone, label, (fX, fY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                          (0, 0, 255), 2)

    # show our detected faces along with smiling/ not smiling labels
    cv2.imshow("Face", frameClone)
    cv2.imshow("Probabilities", canvas)
    if len(rects) > 0:
        cv2.waitKey(3000)

    # # if the 'q' key is pressed, stop the loop
    # if cv2.waitKey(1) & 0xFF == ord("q"):  # attention 0 is zero not O(o)
    #     break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()

# python emotion_detector.py --cascade F:\ProgramPractice\DLFCV\pyimagesearch\sharedresource\haarcascade_frontalface_default.xml --mode
# l F:\data\fer2013\output\checkpoints\weights-071.hdf5 --video F:\data\IMG_4509.MOV

