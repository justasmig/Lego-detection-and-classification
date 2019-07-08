'''

Copyright (c) 2019, justasmig
All rights reserved.

This source code is licensed under the BSD 4-Clause license found in the
LICENSE file in the root directory of this source tree. 

'''





# Importing of libraries.
import cv2
import numpy as np
import tensorflow as tf
import os
import keyboard


# Creating OpenCV video caputure (live camera view). 
# Change "Camera" variables to other numbers if multiple cameras are connected.
Camera = 0
cap = cv2.VideoCapture(Camera) 


# Setting camera exposure to -10 made my view of camera more clear and visible. 
# Experiment with this variable to find your best and clear camera view quality.  
cap.set(cv2.CAP_PROP_EXPOSURE, -10) 


# Reading my labels.txt file to get and use my created labels for lego pieces in Tensorflow. 
# Change directory to labels.txt file if it's placed in other directory.
label_lines = [line.rstrip() for line
               in tf.gfile.GFile('retrained/labels.txt')]


# Defining the function which does graph initialization and reduces logs level to errors only. 
def _init():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Filter out INFO and WARNING logs.

    # Load graph from file, parse it and finally import it to Tensorflow. 
    with tf.gfile.FastGFile('retrained/retrainedLego_graph.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

_init() # Run initialization before starting Tensorflow session and running detection with classification cycle.


# Starting new Tensorflow session.
with tf.Session() as sess:

    # Reading "final_result" tensor of our graph.
    final_result = sess.graph.get_tensor_by_name('final_result:0')

    # Starting detection with classification cycle.
    while True:

        # Read a frame from camera.
        ret, img = cap.read()
        frame = cv2.medianBlur(img,3) # Reduce noise in received frame.

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # Convert frame to HSV format, to remove green background behind object.

        # Lower and upper green value, which will be masked out later. 
        # There might be a need to experiment with these values to mask out green/blue/red background succefully.
        # HSV colors treshold explained here: http://bit.ly/HSV-COLORS-TRESHOLD
        lower_green = np.array([30,40,40]) 
        upper_green = np.array([100,255,255])

        # Creating a mask
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Converting the masked (in this example - green color) part to black.
        im = cv2.bitwise_not(frame,frame, mask= mask)
        im[mask != 0] = [0, 0, 0]

        # Turning parts, that are not green to white and converting image to binary.
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        th, binImg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

        # Creation of Numpy array
        kernel = np.ones((10,10), np.uint8)

        # Making holes in parts (for example motorcycle_wheel) smaller by using our binary image and cv2.MORPH_CLOSE filter.
        binImg = cv2.morphologyEx(binImg, cv2.MORPH_CLOSE, kernel)

        # Finding object's countours in binary image. Smaller holes helps us to find the contour of the whole part, rather it's holes.
        contours,hierarchy = cv2.findContours(binImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 

        # Showing detected object for debugging purposes. 
        # Use this to find out which variables to change and make object detectable. 
        # (i.e. change Numpy array size from 10 to 15 to make holes even smaller, make lower_green/upper_green lower/higher)
        cv2.imshow("Debugging of object detection", binImg)

        # Creating cycle, to classify, draw rectangles and label every object OpenCV has detected.  
        for cnt in contours:            
            x,y,w,h = cv2.boundingRect(cnt) # Object's size and coordinates in image.
            frame = cv2.resize(img[y:y+h, x:x+w], (224, 224), interpolation=cv2.INTER_CUBIC) # Resizing cutted out photo of object, to make it compatible with Tensorflow graph.

            # Converting resized photo of object to numpy array, normalizing it and expanding it to fit image's size.
            numpy_frame = np.asarray(frame)
            numpy_frame = cv2.normalize(numpy_frame.astype('float'), None, -0.5, .5, cv2.NORM_MINMAX)
            numpy_final = np.expand_dims(numpy_frame, axis=0)

            # Feeding image's numpy array to Tensorflow and getting returned unsorted array of predictions of what the detected object is.
            # Example of returned data: nxt_body: 0.8, nxt_motor: 0.05, nxt_sensor: 0.07, wheel_motorcycle: 0.03, ... 
            predictions = sess.run(final_result, {'Placeholder:0': numpy_final})
            
            # Sorting predictions from best to worst.
            top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
            human_string = "" # Temporary variable to put our label in.
            score = 0 # Temporary variable to put our detected object prediction score in.
            
            # Cycling through all of sorted results
            for node_id in top_k:
                # Setting treshold for how big score has to be to draw label.
                if(predictions[0][node_id] > 0.4):

                    #generating and printing label with score in console.
                    human_string = label_lines[node_id]
                    score = predictions[0][node_id]
                    print('%s (score = %.5f)' % (human_string, score))

            # Setting font of label, drawing label above detected object.
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img,human_string,(x,y-20), font, 1,(255,255,255),2,cv2.LINE_AA)

            # Drawing rectangle arround detected object.
            img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            
        # Printing in console showing that our detection session is over.
        print ('********* Session Ended *********')

        # Displaying detected objects on original image.
        cv2.imshow("Detected Objects", img)

        # Press Q button on keyboard to stop detection cycle and quit program.
        # Note: OpenCV will automatically kill detection cycle if there is no "cv2.waitKey(0) command".
        if cv2.waitKey(1) & keyboard.is_pressed('q'): 
            break 

cv2.destroyAllWindows()