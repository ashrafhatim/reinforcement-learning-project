# reinforcement-learning-project-
The final project of the reinforcement learning course during ([AMMI](https://aimsammi.org/)) master 2021. To know more about the project please read the report.

### To download the requirements
```
!pip install gym > /dev/null 2>&1

!pip install gym pyvirtualdisplay > /dev/null 2>&1
!apt-get install -y xvfb python-opengl ffmpeg > /dev/null 2>&1

!apt-get update > /dev/null 2>&1
!apt-get install cmake > /dev/null 2>&1
```
```
!pip install --upgrade setuptools 2>&1
!pip install ez_setup > /dev/null 2>&1
```
```
!pip3 install box2d-py
!pip3 install gym[Box_2D]
```
```
import urllib.request
urllib.request.urlretrieve('http://www.atarimania.com/roms/Roms.rar','Roms.rar')
!pip install unrar
!unrar x Roms.rar
!mkdir rars
!mv HC\ ROMS.zip   rars
!mv ROMS.zip  rars
!python -m atari_py.import_roms rars
```
Or use the following in command line
```
pip install -r src/requirements.txt
```
### The discerete case

#### Parameters
```
parser.add_argument("--env-name", default= "LunarLander-v2", type= str)
    parser.add_argument("--seed", default= 0, type= int)
    parser.add_argument("--gamma", default= 0.99, type= float)
    parser.add_argument("--epsilon", default= 1, type= float)
    parser.add_argument("--epsilon-min", default= 0.01, type= float)
    parser.add_argument("--epsilon-decrement", default= 0.001, type= float)
    parser.add_argument("--learning-rate", default= 0.0001, type= float)
    parser.add_argument("--batch-size", default= 128, type= int)
    parser.add_argument("--n-episodes", default= 100, type= int)
    parser.add_argument("--n-steps", default= 5000, type= int)
    parser.add_argument("--buffer-size", default= 1000000, type= int)
    parser.add_argument("--hid1-dim", default= 200, type= int)
    parser.add_argument("--hid2-dim", default= 128, type= int)
    parser.add_argument("--path", default= "", type= str)
    parser.add_argument("--tb-path", default= "", type= str)
    parser.add_argument("--printLog", default= False, type= bool)
    parser.add_argument("--displayEnv", default= False, type= bool)
```
#### Example
```
python main.py --n-episodes 3 --env-name "LunarLander-v2" --path "../" --tb-path "../" --printLog True
```
### The continuous case
#### Parameters
| option              | description                                                                 | type  | default |
|---------------------|-----------------------------------------------------------------------------|-------|---------|
| --seed              | random seed to be used                                                      | int   | 0       |
| --gamma             | gamma parameter for the algorithm                                           | float | 0.99    |
| --epsilon           | epsilon value for the epsilon greedy actions                                | float | 1       |
| --epsilon-decrement | value by which epsilon is decreased after every episode (linear decay)      | float | 0.001   |
| --epsilon-min       | minimum value for epsilon                                                   | float | 0.1     |
| --learning-rate     | learning rate for the DQN's optimizer                                       | float | 0.0001  |
| --batch-size        | size of batches to be sampled from the replay buffer during each step       | int   | 128     |
| --n-episodes        | number of episodes                                                          | int   | 100     |
| --n-steps           | maximum number of steps for an episode                                      | int   | 5000    |
| --buffer-size       | size of the buffer to store transitions                                     | int   | 1000000 |
| --hid1-dim          | dimension of the first hidden layer                                         | int   | 128     |
| --hid2-dim          | dimension of the second hidden layer                                        | int   | 128     |
| --tb-path           | path of the folder in which the experiment data is saved (tensorboard)      | str   | ""      |
| --path              | path of the folder in which videos of the agent's performance will be saved | str   | ""      |
| --display-env       | whether to display the agent's performance as it learns (done as video)     | bool  | False   |
#### Example
```
python src/discrete/main.py --n-episodes  3 --learning-rate 1e-3  --epsilon-decrement 1e-2 --batch-size 512 --path "../../" --tb-path "../../"
```
### Part of the results
#### LunarLander-v2
![alt text](https://github.com/ashrafhatim/reinforcement-learning-project-/blob/master/images/plot1.png)
![alt text](https://github.com/ashrafhatim/reinforcement-learning-project-/blob/master/images/plot2.png)
![alt text](https://github.com/ashrafhatim/reinforcement-learning-project-/blob/master/images/plot3.png)

#### LunarLanderContinuous-v2
![alt text](https://github.com/ashrafhatim/reinforcement-learning-project-/blob/master/images/plot4.png)
![alt text](https://github.com/ashrafhatim/reinforcement-learning-project-/blob/master/images/plot5.png)
![alt text](https://github.com/ashrafhatim/reinforcement-learning-project-/blob/master/images/plot6.png)
