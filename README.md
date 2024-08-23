# PDS_Compute_MTF x mtf.py

This repository is a combination of [PDS_Compute_MTF](https://github.com/bvnayak/PDS_Compute_MTF) and [mtf.py](https://github.com/u-onder/mtf.py) for MTF (Modular Transfer Function) analysis of an image. Furthermore, new features are implemented that allow the processing of video streams.

## Features

* Uses PDS_Compute_MTF to allow the user to select a rectangular region from an input image that should be analyzed
* Uses mtf.py to do edge detection on an image cutout and to calculate the MTF (mtf.py is stored in the directory ``mtf``)

Newly added features:
* Edited mtf.py such that it calculates and visualizes the MTF50 value from the MTF
* Edited the image cutout selection of PDS_Compute_MTF such that it works on video streams instead of single image files

## Set-Up with Conda

```
conda create -y --name MTF python==3.10.12
conda activate MTF
pip install -r requirements.txt
```

## How to Use
Run the program via:
```
python -m PDS_Compute_MTF.run_mtf_analysis
```
A window appears that shows the video stream from the capture device defined in PDS_Compute_MTF/run_mtf_analysis.py.

To do MTF analysis:
1. Double click on the video stream. The video will be paused.
2. Use your mouse to select a rectangular region from the image. The region will be marked red.
3. Press 'Enter' on your keyboard.
4. A new window will open that shows the MTF analysis. This window will stay open and will always contain the latest analysis. Even if you resume the video.
5. Double click on the video again to resume it.
6. To do another analysis repeat steps 1-4. This will overwrite the latest analysis. Save your results beforehand.
