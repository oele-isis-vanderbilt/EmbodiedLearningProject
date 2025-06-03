# Offical Repo for 3D Gaze Tracking in Embodied Learning

Within this repo, the code required to perform 3D gaze, including depth estimation, gaze estimation, 3d reconstruction, and gaze ray tracking can be found. The 2D gaze tracking method can be found [here](https://github.com/oele-isis-vanderbilt/GazeAnalysisLI).

# Repo Data

[Project data stored in Vanderbilt Box](https://vanderbilt.box.com/s/e2v2mlsz2qgc2mosvghiwfue4pp1k713)

This repo includes the source code and the data for the [AIED2024 submission](https://link.springer.com/chapter/10.1007/978-3-031-64299-9_1). Store the data located in the Vanderbilt Box shared folder within the empty ``data`` directory at the root of the Github repo. This ``data`` folder in the Vanderbilt box includes generated artefacts from the scripts along with human annotations/corrections that are necessary for runnign the scripts.

## Preprocessing Step

Once finished downloading the data located in the Vanderbilt box, please unzip the following files:

```
reid/cropped_faces/d1g1.zip -> reid/cropped_faces/d1g1/
reid/cropped_faces/d1g2.zip -> reid/cropped_faces/d1g2/
reid/cropped_faces/d2g1.zip -> reid/cropped_faces/d2g1/
reid/cropped_faces/d2g2.zip -> reid/cropped_faces/d2g2/
```

This was required for uploading the raw directories resulted in failed uploads.

# Script Organization and Purpose

The gaze pipeline within the ``scripts/gaze`` directory is composed of the following linear sequence:
1. ``reid.ipynb``: We started with a semi-automated ReID (a precursor of the ReID project in the OELE lab) that uses facial recognition, from the ``deepface`` [PyPI package](https://pypi.org/project/deepface/), that matches the face with the larger body bounding boxes. Manual corrections were needed, since facial recogntion was error prone with children participants.
2. ``depth_estimation.ipynb``: Used [ZoeDepth](https://github.com/isl-org/ZoeDepth) to perform metric depth estimation on the entire RGB videos and generated depth videos.
3. ``gaze_estimation.ipynb``: Used [L2-CSNet](https://github.com/Ahmednull/L2CS-Net) to perform gaze estimation on the videos with face crops.
4. ``reconstruction3d.ipynb``: This is the bulk of the work, including the 3D gaze and Object-Of-Interest encoding. This file performs a ``process`` that takes the tracking, video, depth, and gaze and reconstructs the entire 3D scene as a point cloud, 3D bounding boxes, and 3D gaze rays. Using a custom 3D video plotter named [Plot3D](https://github.com/edavalosanaya/Plot3d), which could be replaced with the more reliable Open3D Visualizer, we could visualize the reconstructed 3D scene and its progression throughout the session. Within this code, we perform the following: 
    - Use human-annotated positions of the floor and projector, using the [Vision6D](https://github.com/InteractiveGL/vision6D) tool to manually place these objects in the 3D scene.
    - Using the depth video, place the 3D gaze vector and bounding box matching each participant.
    - Perform gaze ray tracking using Trimesh to identify what person or object a participant is looking at.
    - Display the 3D scene with Plot3D and tag each video frame with PersonA->Object/PersonB information.
5. ``final_merge.ipynb``: Lastly, this script meets the requirement of the output format -- time-window, pooled at N seconds, and the gaze target IDs match the human-readable ID (e.g., Taylor Swift instead of Student1).

# Future Improvement Suggestions

For future data analysis, I recommend the following suggestions: 
1. **ReID.** The ReID pipeline needs to be updated to take advantage of Ashwin et al.'s newer and more powerful ReID pipeline.
2. **Depth.** Use the depth stereo cameras from Luxonis such as the Oak-D Pro W that were recently purchased by the OELE lab.
3. **3D Video Plotting.** Instead of using the poorly documented and custom-made Plot3D, I recommend using the better established [Open3D Visualizer](https://www.open3d.org/docs/latest/python_api/open3d.visualization.Visualizer.html), with non-blocking visualization to support the real-time updates from video -- [docs](https://www.open3d.org/docs/release/tutorial/visualization/non_blocking_visualization.html) found here.
4. **Gaze Estimation.** L2-CSNet is convenient and easy to use -- however; it's performance is lacking and there is likely to be a much better performing gaze estimation method available now.
