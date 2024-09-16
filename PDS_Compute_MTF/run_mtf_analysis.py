import numpy as np
import cv2
import argparse
import time
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import RectangleSelector
from matplotlib.backends.backend_pdf import PdfPages
from scipy import interpolate
from scipy.signal import savgol_filter

from mtf import mtf


class EventHandler(object):

    def __init__(self):
        self.image_array = None
        # shows a text below the figure with the coordinates of the current rectangle
        self.coordinates_text = plt.figtext(0.5, 0.01, "", ha="center")
        
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
        self.coordinates_text.set_text(f"ROI coordinates: ({int(x1)}, {int(y1)}), ({int(x2)}, {int(y2)})")
        plt.draw()

    def key_pressed(self, event):
        if event.key in ['enter']:
            # Run MTF analysis
            fig = plt.figure(2, figsize=(9, 6)) # write to figure 2 which should always show the current mtf results
            fig.clear()
            plt.connect('key_press_event', self.key_pressed) # such that the new window inherits the key events of the main window
            image_cropped = self.image_array[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]
            print("Running MTF analysis")
            mtf_result = mtf.MTF.CalculateMtf(image_cropped, verbose=mtf.Verbosity.DETAIL) # CHANGE TO Verbosity.DETAIL to show plots!
        elif event.key == 'q':
            # close program
            global close_program
            close_program = True
            print("Key 'q' was pressed. Closing the program")
        elif event.key == 'ctrl+s':
            save_all_figures_to_pdf()


def save_all_figures_to_pdf(filename_without_extension=None, directory='results'):
    #https://stackoverflow.com/questions/26368876/saving-all-open-matplotlib-figures-in-one-file-at-once
    os.makedirs(directory, exist_ok=True)
    filecounter_path = os.path.join(directory, 'filecounter.data')
    if not os.path.exists(filecounter_path):
        with open(filecounter_path, 'x') as f:
            f.write('1')
            filecounter = 1
    else:
        with open(filecounter_path, 'r') as f:
            filecounter = int(f.read())

    if filename_without_extension:
        filename = str(filecounter) + "_" + filename_without_extension + ".pdf"
    else:
        filename = str(filecounter) + ".pdf"

    with PdfPages(os.path.join(directory, filename)) as pdf:
        figs = [plt.figure(n) for n in plt.get_fignums()]
        for fig in figs:
            fig.savefig(pdf, format='pdf')

    filecounter += 1
    with open(filecounter_path, 'w') as f:
        f.write(str(filecounter))

    print(f"Saved the current figures to {filename}. The above expcetion (AttributeError: 'FigureCanvasPdf' object has no attribute 'copy_from_bbox') can be ignored. This happens all the time and I cannot catch this exception.")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--roi', nargs=4, type=int, 
                        help='x1,y1 (top left) and x2,y2 (bottom right) coordinates of the roi in the format: "--roi x1 y1 x2 y2"')
    args = parser.parse_args()
    fixed_roi = False
    x1, y1, x2, y2 = [-1]*4
    if args.roi:
        fixed_roi = True
        x1, y1, x2, y2 = args.roi
    cap = cv2.VideoCapture('/dev/video0')
    fig_image, current_ax = plt.subplots()
    fig_image.canvas.manager.set_window_title('Live Video')

    eh = EventHandler()
    plt.connect('key_press_event', eh.key_pressed)
    if fixed_roi:
        eh.roi = [y1, y2, x1, x2]
    else:
        rectangle_selector = RectangleSelector(current_ax,
                                               eh.line_select_callback,
                                               useblit=True,
                                               button=[1, 2, 3],
                                               minspanx=5, minspany=5,
                                               spancoords='pixels',
                                               interactive=True)
        plt.connect('button_press_event', eh.mouse_clicked)

    # instead use ctrl+s for my own save function that saves all open figures
    plt.rcParams['keymap.save'].remove('ctrl+s')
    
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    
    video_paused = False
    close_program = False
    video_paused_changed = False
    last_click_time = 0
    
    print("Double click on the video to pause and select a rectangle")
    
    while(not close_program):
        if close_program:
            plt.close('all')
            exit()
        if not video_paused: 
            ret, frame = cap.read()
            if ret == True:
                 #cv2.imshow('Frame',frame)
                 print("new video frame")
                 plt.figure(1, figsize=(9, 6))
                 plt.title("close program by pressing 'q'")
                 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) / 255
                 plt.imshow(frame, cmap='gray')
                 if fixed_roi:
                    rectangle_width = np.abs(x2-x1)
                    rectangle_height = np.abs(y2-y1)
                    plt.gca().add_patch(
                        mpatches.Rectangle(
                            (x1,y1),rectangle_width,rectangle_height,
                            linewidth=1,edgecolor='r',facecolor='none'
                        )
                    )
                    plt.suptitle("Using a fixed ROI")
                 else:
                    plt.suptitle("Select ROI with the mouse")
                 
                 eh.image_array = frame
                 
                 
                 # without block=False the loop is paused until we close the plot
                 plt.show(block=False)
                 #plt.pause(0.001)
                 
                 #if plt.waitforbuttonpress(0.001):
                 #    break
                 
                 plt.pause(0.0001)
                 
            else:
                break
        else:
            plt.pause(0.0001)
   	   
    	
