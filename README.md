# Thesis
Repo for my master's thesis @ UIC and Politecnico di Milano

***

The goal of this thesis work is to provide ophthalmologists with additional real-time, more reliable information of the
distance of the instrument from the retina to avoid damages to patients.

The approach chosen is to use Deep Neural Networks (ConvNets) to locate the instrument's tip in the surgeries videos and
Stereo Vision to estimate its distance from the retina.

The repo is divided as follows.
<pre>
~/
| -- data/
|     |
|     | -- datasets/
|     | -- outputs/
|     | -- targets/
|     | -- videos/
|     | -- videos_2D/
|
| -- source/
      |
      | -- GUI/
      | -- disparity_scripts/
      | -- helper_scripts/
      | -- tool_detection/
      | -- main.py
      | -- main-cnn-only.py

</pre>
Under the **data** sub-folder you can find:
- the 2D and 3D surgery videos (not pushed remotely for size concern) respectively in **video_2D** and **videos** 
- the 2D and 3D frames extracted from the videos (not pushed remotely) under **datasets**
- the annotation files, i.e. the targets for the ConvNet under **targets**

Under the **source** sub-folder you can find:
- _GUI_ containing the source code of the marking tools used to annotate the frames to train the CNNs.
- _disparity scripts_ containing the scripts used to test disparity computation functions and methods
- _tool detection_ containing the code for the detection of the tool, including Optical Flow scripts, ORB scripts and 
CNNs training & testing scripts
- _helper scripts_ containing all the helper scripts, including scripts to extract, resize, crop the frames from the videos,
 to check annotations on the frames and so on.

The **source** sub-folder also contains the _main-xxx.py_ scripts that implement a particular version of the entire pipeline.