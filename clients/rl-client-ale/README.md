# Client for Atari games (ALE)
 
 Client for Arcade-Learning-Environment emulator, which allows to launch Atari games.
 
 Just run `main.py` which already has default parameters
 or change some via command line.
 
 Example > a3c algorithm with 8 parallel agents:
 > main.py --agents 8
 
 You can also specify the game name (boxing for default) and algorithm:
 > main.py --game pong --algo dqn
 
 If you want to see some visual output for one game just type in terminal `d` then `Enter`
 (if game didn't start try to repeat one more time)
 
## How to Run & Dependencies

Before you start, make sure you have installed on your system:

- `python 2.7 or 3.5`

- [`pip`](https://pip.pypa.io/en/stable/installing/) - just need to install requirements, see command below:
    > pip install -r requirements.txt
    
- [Arcade Learning Enviroment](https://github.com/4SkyNet/Arcade-Learning-Environment) - see commands below:

`$ git clone https://github.com/4SkyNet/Arcade-Learning-Environment`

`$ cd Arcade-Learning-Environment`

`$ cmake -DUSE_SDL=ON -DUSE_RLGLUE=OFF -DBUILD_EXAMPLES=OFF .`

`$ make -j 4` (4 - number of cores, which you have or want to use)
	
`$ pip install .`

If installation of Atari Environment failed try to install these dependencies:

`sudo apt-get install libsdl1.2-dev libsdl-gfx1.2-dev libsdl-image1.2-dev cmake`

[ALE Manual](https://github.com/mgbellemare/Arcade-Learning-Environment/blob/master/doc/manual/manual.pdf)


#### [How to create a virtual environment](/VirtualEnvironments.md)