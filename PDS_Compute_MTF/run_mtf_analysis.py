import numpy as np
import cv2
import argparse
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

    def __init__(self, filename):
        self.filename = filename

    def line_select_callback(self, eclick, erelease):
        'eclick and erelease are the press and release events'
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        self.roi = np.array([y1, y2, x1, x2])
        # round roi to full pixels
        self.roi = np.round(self.roi) 
        self.roi = self.roi.astype(int)

    def event_exit_manager(self, event):
        if event.key in ['enter']:
            #PDS_Compute_MTF(self.filename, self.roi)
            img_array = mtf.Helper.LoadImageAsArray(filename)
            fig = plt.figure(2) # write to figure 2 which should always show the current mtf results
            fig.clear()
            mtf_result = mtf.MTF.CalculateMtf(img_array[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]], verbose=mtf.Verbosity.DETAIL)

class ROI_selection(object):

    def __init__(self, filename):
        self.filename = filename
        self.image_data = cv2.imread(filename, 0)
        fig_image, current_ax = plt.subplots()
        plt.imshow(self.image_data, cmap='gray')
        eh = EventHandler(self.filename)
        rectangle_selector = RectangleSelector(current_ax,
                                               eh.line_select_callback,
                                               useblit=True,
                                               button=[1, 2, 3],
                                               minspanx=5, minspany=5,
                                               spancoords='pixels',
                                               interactive=True)
        plt.connect('key_press_event', eh.event_exit_manager)
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', help='String Filepath')
    args = parser.parse_args()
    filename = args.filepath
    ROI_selection(filename)
