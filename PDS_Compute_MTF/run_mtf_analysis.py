import numpy as np
import cv2
import argparse
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import RectangleSelector
from scipy import interpolate
from scipy.signal import savgol_filter

from mtf import mtf

# Reference:
# http://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]


class EventHandler(object):

    def __init__(self):
        self.image_array = None
        
    def mouse_clicked(self, event):
        global video_paused
        if event.dblclick:
            video_paused = not video_paused
            if video_paused:
                print("Video paused.")
            else:
                print("Video resumed")

    def line_select_callback(self, eclick, erelease):
        'eclick and erelease are the press and release events'
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        self.roi = np.array([y1, y2, x1, x2])
        # round roi to full pixels
        self.roi = np.round(self.roi) 
        self.roi = self.roi.astype(int)

    def key_pressed(self, event):
        if event.key in ['enter']:
            # Run MTF analysis
            fig = plt.figure(2) # write to figure 2 which should always show the current mtf results
            fig.clear()
            image_cropped = self.image_array[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]
            print("Running MTF analysis")
            mtf_result = mtf.MTF.CalculateMtf(image_cropped, verbose=mtf.Verbosity.DETAIL)
        elif event.key == 'q':
            # close program
            global close_program
            close_program = True
            print("Key 'q' was pressed. Closing the program")
    

if __name__ == '__main__':
    #parser = argparse.ArgumentParser()
    #parser.add_argument('filepath', help='String Filepath')
    #args = parser.parse_args()
    #filename = args.filepath
    #ROI_selection(filename)
    cap = cv2.VideoCapture('/dev/video0')
    fig_image, current_ax = plt.subplots()
    fig_image.canvas.manager.set_window_title('Live Video')

    eh = EventHandler()
    rectangle_selector = RectangleSelector(current_ax,
                                           eh.line_select_callback,
                                           useblit=True,
                                           button=[1, 2, 3],
                                           minspanx=5, minspany=5,
                                           spancoords='pixels',
                                           interactive=True)
    plt.connect('key_press_event', eh.key_pressed)
    plt.connect('button_press_event', eh.mouse_clicked)
    
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    
    video_paused = False
    close_program = False
    video_paused_changed = False
    last_click_time = 0
    
    print("Double click on the video to pause and select a rectangle")
    
    while(not close_program):
        if close_program:
            print("Close program is true")
            exit()
        if not video_paused: 
            ret, frame = cap.read()
            if ret == True:
                 #cv2.imshow('Frame',frame)
                 print("new video frame")
                 plt.figure(1)
                 plt.title("close program by pressing 'q'")
                 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) / 255
                 plt.imshow(frame, cmap='gray')
                 
                 eh.image_array = frame
                 
                 
                 # without block=False the loop is paused until we close the plot
                 plt.show(block=False)
                 #plt.pause(0.001)
                 
                 #if plt.waitforbuttonpress(0.001):
                 #    break
                 
                 plt.pause(0.001)
                 
            else:
                break
        else:
            plt.pause(0.001)
   	   
    	
