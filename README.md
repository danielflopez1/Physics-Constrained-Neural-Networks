# Physics learning using neural networks
In this [Research](https://drive.google.com/file/d/1NNsIXCzIrc_wk570KmOr1ck69hq83zE1/view?usp=sharing) we show how the principle of spatio-temporal locality, inherent in all physical laws, can be used to create highly data-efficient deep learning architectures capable of embodying/discovering various elements of physical laws. We design the networks to understand and predict the time evolution of a system of colliding balls in a box in a gravity field. We show that learning from just one video less than a few seconds long, our networks can predict, with high accuracy, the evolution of unseen systems characterized by a different number of balls, different sized/shaped boxes, etc. Second, we design networks to understand the dynamics of a system governed by the Burrdge-Knopoff equations – idealization of earthquake propagation. In this, we show that from just one simulation, our networks can predict, with high accuracy, the generation of earthquake events in unseen systems characterized by different material properties of the medium in which the earthquakes are propagating. 

## How to use
### Prerequisites
```
Tensorflow
Keras
Pygame 
Pymunk
Numpy 
h5py
```

# Configuration
Every file is independent of each other, however, you may run in this sequence:
- [simulator.py](#Simulator)
- [box_select.py](#BoxSelect)
- [cnn_lstm_training.py](#CNN LSTM training)
- [time_marching.py](#Time Marching)

### Simulator
 simulator.py generates a physics simulation frames of a tracked object, square or polygon boundaries and other objects. 
#### Inputs
```
sim = Simulate(scenario = 4, generate_data = True, take_video = False, file_name = 'filename')
sim.main()
```

#### Scenarios
1 - tracked ball in projectile motion.
2 - tracked ball in square boundaries.
3 - tracked ball and another ball in square boundaries.
4 - tracked ball and 10 balls in a square with modifiable barrier size.
5 - tracked ball with box objects in square boundaries.
6 - tracked ball with boxes in square boundaries.
7 - tracked ball with different polygon boundaries. 
8 - tracked ball and 10 balls with a modifiable polygon. 
9 - tracked ball and 10 balls and a simpler polygon for testing - this was constantly modified.

#### Outputs
If generate_data is True, it will generate HDF5 files with frames of data and an HDF5 file with the coordinates and rotation of the tracked ball.
If take_video is True, it will make a video of the frames.

### BoxSelect
boxSelect.pyCrops the HDF5 frames using the HDF5 coordinates file which generates Locality.
#### Inputs
```
crop = SelectImage(size=100, frame_file = "full_ball_in_square_sider_v3.h5",coordinate_file = "xyrot_ball_in_square_sider_v3.h5")
crop.output()
```
#### Outputs
Cropped HDF5 file of stated size.

### CNN LSTM training
cnn_lstm_training.py modifies the input files in the file and run the file 
#### Inputs
Take multiple corresponding cropped HDF5 frame files and HDF5 coordinate file
input_filename1 = '/home/username/datasets-1.h5'
output_filename2 = '/home/username/xydatasets-1.h5'
....
input_filenameN = '/home/username/datasets-N.h5'
output_filenameN = '/home/username/xydatasets-N.h5'

#### Outputs
Saved model

### BlankSlate
BlankSlate.py generates a blank slate by removing the previous red ball and making a new ball in the new coordinate.
This is used in time_marching.py to generate a frame from a prediction of the model. 
#### Inputs
Tkes size of the cropped frame, HDF5 frames file, frame index and and the predicted coordinate
```
img = changeFrame(size = 100, frame_file = 'full_3ball_in_concave_polygon_v3.h5')
img.output(frame_index,x,y)
```
#### Output
It outputs a frame 2Darray (It is not HDF5 since it is being used in time_marching)

### Time Marching
time_marching.py is modified and then ran in python. It uses a starting HDF5 file to generate predictions that are set as a new 2D frame and queues its image predictions as inputs to generate predictions over time.
#### Inputs
It uses a the HDF5 trained network, input and output normalizing file used in training and the full and cropped frame of the file of which you want to generate predictions.

trainedNet_file = '/home/trained_net.h5'
input_filename1_norm = '/home/datanormalizer_input.h5'
output_filename1_norm = '/home/xydatanormalizer_input.h5'
test_set1 = '/home/fullframe_to_use_blankslate.h5'
inp2 = '/home/cropped_frames_to_predict_over.h5'

#### Outputs
Coordinate predictions over X timesteps.

# Developers
* Daniel Lopez 
* David Finol
* Ankit Srivastava
