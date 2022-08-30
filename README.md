# ourVO
## Image Retrieval
Use my_feature_extract.py extract the descriptor of each images in training set for image retrieval.
```angular2html
python my_feature_extract.py
```
Modify the images path and save path as you need.
The data structure of your retrieval database is supposed to be:
```angular2html
----Site A
    ----seq x
        ----db(for images data)
        ----output_feature(for the descriptor extracted by patchnetvlad)
        ----pose_unit.py(converted groundtruth for visual odometry)
```
In order to get the pose_unit.py file, you should run posefile_handle_unit.py, feel free to change the path as you need.
## Start VO
Run visual_odometry_wanglong.py to start the tracking process
```angular2html
python visual_odometry_wanglong.py
```
You may need to change the relative path for adjust your prepared data.
SuperPoint and SuperGlue model file is also needed in the projects, please download by yourself.

