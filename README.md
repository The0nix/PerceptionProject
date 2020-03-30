# PerceptionProject
This is the repository for the final project in the Perception in Robotics Skoltech course.

Authors: Tamerlan Tabolov, Alexey Vyatkin, Anton Voronov and Kazii Botashev. 

## Usage
To launch the algorithm just run `main.py`:
```
python main.py -a
```
The folowing arguments are suppported:
* `-a, --animate` — show animation of trajectory
* `-i INPUT, --input INPUT` — input dataset path
* `-o OUTPUT, --output OUTPUT` —file to output trajectory to
* `-s SKIP_FRAMES, --skip-frames SKIP_FRAMES` — how many IMU measurements to skip at each step 
