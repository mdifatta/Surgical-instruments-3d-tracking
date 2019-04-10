# Thesis
Repo for my master's thesis @ UIC

The repo is divided into **data** and **source** sub-folders.

The **data** sub-folder contains:
- the raw medical videos (not pushed remotely)
- the test outputs (not pushed remotely)
- the target files
- the 2D and 3D frames extracted from the videos (not pushed remotely)

The **source** sub-folder contains the code of the 4 sub-projects:
- _GUI_ contains the source code of the marking tool used to mark the frames and create the training dataset for the NNs
- _stereo depth_ contains all the scripts for disparity and depth computation
- _tool shadow_ contains all the scripts for the detection of the tool's and shadow's ends and for the computation of their distance
- _video handler_ contains all the scripts for the extraction of frames from the 2D and 3D videos