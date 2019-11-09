# Thesis
Repo for my master's thesis @ UIC

The repo is divided into **data** and **source** sub-folders.

The **data** sub-folder contains:
- the 2D and 3D surgery videos (not pushed remotely for size concern)
- the 2D and 3D frames extracted from the videos (not pushed remotely)
- the annotation files, i.e. the targets for the ConvNet

The **source** sub-folder contains the code of the four sub-projects:
- _GUI_ contains the source code of the marking tool to annotate both the tool's tip and the shadow and create the targets for the first version of the CNN
- _GUIv2_ contains the source code of the marking tool to annotate ONLY tool's tip and the shadow and create the targets for the CNN
- _stereo depth_ contains all the code for disparity and depth computation
- _tool shadow_ contains all the code for the detection of the tool, the shadow and for the computation of their distance
- _video handler_ contains all the scripts for the extraction and processing of 2D and 3D frames

The **source** sub-folder also contains the _main-xxx.py_ scripts that implement a particular version of the entire pipeline.