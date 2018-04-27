# Control theory 
## Assignment#5
### Results
The best result shown by sensor fusion (odometer and gyroscope). Gyroscope by himselft isn't too precise. To measure with him I took average speed of both wheels and angle, assume that robot moves in that direction. Nevertheless it works. I can not apply Kalman's filter to results of this estimation but I did apply to sensor's data directly. Only after that I handle x and y values. 

Better results shown by odometer, it has more accuracy. Also, I can't apply Kalman's filter to results of measurements. Because of its non-linereaty.

Woth to note, in all cases I used a *magic_coefficient* to transform odometer's data into velocity. It equals 0.055 which near to the 'real' value 2\*pi\*2.7/360~0.047

### Following by odometer
![alt text](https://github.com/prokop7/diff-drive-robot-trajectory/blob/master/plots/wheels.png "By odometer")
### Following by gyroscope
![alt text](https://github.com/prokop7/diff-drive-robot-trajectory/blob/master/plots/gyro.png "By gyroscope")
### Following by gyroscope with applying Kalman's filter
![alt text](https://github.com/prokop7/diff-drive-robot-trajectory/blob/master/plots/gyro%20with%20kalman.png "By gyroscope with Kalman")
### Real location from camera
![alt text](https://github.com/prokop7/diff-drive-robot-trajectory/blob/master/plots/camera.png "From camera")
### Real location from camera with Kalman's filter
![alt text](https://github.com/prokop7/diff-drive-robot-trajectory/blob/master/plots/camera%20with%20kalman.png "From camera with Kalman's filter")
### Following with sensor fusion (odometer and gyroscope)
![alt text](https://github.com/prokop7/diff-drive-robot-trajectory/blob/master/plots/kalman%20with%202%20sensors.png "By gyroscope and odometer")
### Unsuccessful particle filter (distance to wall)
![alt text](https://github.com/prokop7/diff-drive-robot-trajectory/blob/master/plots/particle%20filter.png "Partical filter")
