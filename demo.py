# A demonstration of how image processing will be used for project

import numpy as np
import cv2
from matplotlib import image as image
import easygui
import tkinter as tk
import os.path
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot

def upload():
    # open file explorer to select image
    f = easygui.fileopenbox()

    #set I as global variable that can be passed to different function
    global I

    # read image and store in I
    I = cv2.imread(f)


def segment():
    pass

    # Gets the I from where it was previously used
    global I

    #copy to im var
    im = I.copy()

    # convert rgb to hsv  for thresholding reasons due to more control with hue
    HSV = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

    #trackbar does nothing and simply pass the x position of the slider
    def nothing(x):
        pass
    cv2.namedWindow('Threshold')

    wnd = 'Threshold'
    #Creating a trackbar for the lower hue,saturation and value, and one upper for hue
    #nothing is the call back funciton which is executed everything the trackbar position changes
    cv2.createTrackbar("Lower_Hue", "Threshold",0,179,nothing)     #lower hue
    cv2.createTrackbar("Lower_Sat", "Threshold",35,255,nothing)     #lower sat
    cv2.createTrackbar("Lower_Val", "Threshold",100,255,nothing)     #lower val
    cv2.createTrackbar("Upper_Hue", "Threshold",100,255,nothing)     #upper hue

    while(1):
        Lower_Hue=cv2.getTrackbarPos("Lower_Hue", "Threshold")
        Lower_Sat=cv2.getTrackbarPos("Lower_Sat", "Threshold")
        Lower_Val=cv2.getTrackbarPos("Lower_Val", "Threshold")
        Upper_Hue=cv2.getTrackbarPos("Upper_Hue", "Threshold")

        # set the threshold to cover the unwanted part using trackbar
        global thresh
        thresh =cv2.inRange(HSV,(Lower_Hue,Lower_Sat,Lower_Val),(Upper_Hue,255,255))
        
        
        cv2.imshow("threshold", thresh)

        #exits by pressing q
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        elif k == 27:
            break

    #Shows the region of interest only on a white background
    global ROI
    ROI = cv2.bitwise_and(I,I,mask=thresh)

    #convert black pixels to white as white background
    for i in range(ROI.shape[0]):
        for j in range(ROI.shape[1]):
            if ROI[i,j,0]==0 and ROI[i,j,1]==0 and ROI[i,j,2]==0:
                ROI[i,j,:] = 255
    
    global img
    img = ROI
    cv2.imshow("Final Image",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

def transform():
    global img

    # convert to numpy array
    data = img_to_array(img)

    # expand dimension to one sample
    samples = expand_dims(data, 0)
    print(samples)

    # create image data augmentation generator
    datagen = ImageDataGenerator(width_shift_range=[-50,100])

    # iterator
    it = datagen.flow(samples, batch_size=1)

    # generate samples and plot
    for i in range(9):
        # define subplot
        pyplot.subplot(330 + 1 + i)
        # generate batch of images
        batch = it.next()
        # convert to unsigned integers for viewing
        image = batch[0].astype('uint8')
        # plot raw pixel data
        pyplot.imshow(image)
    # show the figure
    pyplot.show()

############## Creating buttons and setting size of UI window
   
HEIGHT = 700
WIDTH = 700

root = tk.Tk()
root.wm_title("Segment Leaf")

canvas = tk.Canvas(root, height=HEIGHT, width=WIDTH)
canvas.pack()

frame = tk.Frame(root,bg='#dbf4fc', bd=5)
frame.place(relx=0.5, rely=0.1, relwidth=0.75, relheight=0.75, anchor='n')


entry = tk.Entry(frame, font=40)
entry.place(relwidth=1, relheight=0.1)

button1 = tk.Button(frame, text="Upload image", font=40, command = upload)
button1.place(relx=0.2, rely=0.3, relheight=0.1, relwidth=0.5)

button2 = tk.Button(frame, text="Segment", font=40, command = segment)
button2.place(relx=0.2, rely=0.5, relheight=0.1, relwidth=0.5)

button3 = tk.Button(frame, text="Transform", font=40, command = transform)
button3.place(relx=0.2, rely=0.7, relheight=0.1, relwidth=0.5)


root.mainloop()