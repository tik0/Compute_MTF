# PDS_Compute_MTF x mtf.py

This repository is a combination of [PDS_Compute_MTF](https://github.com/bvnayak/PDS_Compute_MTF) and [mtf.py](https://github.com/u-onder/mtf.py) for MTF (Modulation Transfer Function) analysis of an image. Furthermore, new features are implemented that allow the processing of video streams.

## Features

* Uses PDS_Compute_MTF to allow the user to select a rectangular region from an input image that should be analyzed
* Uses mtf.py to do edge detection on an image cutout and to calculate the MTF (mtf.py is stored in the directory ``mtf``)

Newly added features:
* Edited mtf.py such that it calculates and visualizes the MTF50 value from the MTF
* Edited mtf.py such that it can analyze curved edges by approximating the edge with a polynomial function
* Edited the image cutout selection of PDS_Compute_MTF such that it works on video streams instead of single image files
* Added a fixed image cutout selection via CLI parameters
* Added functionality to save the analysis results to a PDF

## Set-Up with Conda

```
conda create -y --name MTF python==3.10.12
conda activate MTF
pip install -r requirements.txt
```

## How to Use
This tool shows the camera image and lets you select a region of interest (ROI) for which you would like to perform a MTF analysis. You can either select the ROI manually with the mouse or you can pass a fixed region of interest via a CLI parameter such that multiple MTF analyses have the same ROI.

### Select a region of interest manually 

Run the program via:
```
python3 -m PDS_Compute_MTF.run_mtf_analysis
```

To do MTF analysis:
1. Double click on the video stream. The video will be paused.
2. Use your mouse to select a rectangular region from the image. The region will be marked red.
3. Press 'Enter' on your keyboard.
4. A new window will open that shows the MTF analysis. This window will stay open and will always contain the latest analysis. Even if you resume the video.
5. Double click on the video again to resume it.
6. To do another analysis repeat steps 1-4. This will overwrite the latest analysis. Save your results beforehand. To save the results to a PDF press `CTRL+S`. The PDF will be saved in a new directory called `results`.

### Use a fixed region of interest

Run the program via:
```
python3 -m PDS_Compute_MTF.run_mtf_analysis --roi x1 y1 x2 y2
```
Example:
```
python3 -m PDS_Compute_MTF.run_mtf_analysis --roi 740 790 1400 1100
```
Whereas x1 and y1 are the coordinates of the top left corner of the ROI. Similarly, x2 and y2 are the coordinates of the bottom right corner.

In order to find suitable coordinates for the ROI you can firstly run the tool with `python3 -m PDS_Compute_MTF.run_mtf_analysis`. Then select a region with the mouse and use the coordinates shown below the camera image as fixed CLI parameters.

Once you run the tool with fixed ROI coordinates, here is how you do an MTF analysis:
1. Press 'Enter' on your keyboard.
2. A new window will open that shows the MTF analysis. This window will stay open and will always contain the latest analysis.
3. To save the results of the analysis to a PDF press `CTRL+S`. The PDF will be saved in a new directory called `results`.
5. Once you changed the configuration of your camera, press 'Enter' again to perform a new analysis.
