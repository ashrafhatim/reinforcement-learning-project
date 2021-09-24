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
