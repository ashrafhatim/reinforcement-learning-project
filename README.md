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
```
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
    parser.add_argument("--displayEnv", default= False, type= bool)
```
#### Example
```
python main.py --n-episodes  3 --learning-rate 1e-3  --epsilon-decrement 1e-2 --batch-size 512 --path "../../" --tb-path "../../"
```

