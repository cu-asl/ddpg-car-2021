# ddpg-car-2021
## 2103499 MECH ENG PROJECT
## Department of Mechanical Engineering, Faculty of Engineering, Chulalongkorn University
## A 2021-senior project under the supervision of Asst. Prof. S. Chantranuwathana, Ph.D. and R. Tse, Ph.D.
## Contributed by P. Pongsuwannakul, R. Sutaporn, P. Tangsripairoje

Written by P. Tangsripairoje
### Getting started
This project is a reinforcement learning-based car simulation which simulated on the CARLA simulator (v0.9.12) and implemented on Python 3.7.
The algortithm we used is *Deep Deterministic Policy Gradient (DDPG)*

**Device specifications**  

OS Recommedation: Windows 10 or 11. *Ubuntu is fine* but **Mac OS is not allowed**.

Installed RAM: Should be at least 16.0 GB

Installed GPU: Only NVIDIA supports CARLA so AMD is not allowed. Also, 8.0 GB-RAM GPU is recommended. (Even though, I used the 4.0 GB one but still works fine.)

You also need the CARLA simulator v0.9.12 (I don't recommend using the versions below this because there are major changes in every version. Moreover, if you want to use a higher-version one, there are also risks that some further changes might affect the code.)

**CARLA (All versions)**: https://github.com/carla-simulator/carla/releases


Then you need Python 3.7 (This project used Python 3.7.9, but 3.7.x also work fine.)

**Python 3.7.9**: https://www.python.org/downloads/release/python-379/

Finally, you need to install the dependencies which is stored in `requirements.txt`. 
If you are using pip, you can simply install by typing this command in the command prompt.

```
pip install requirements.txt
```
If you don't have pip, just go to [this website](https://bootstrap.pypa.io/get-pip.py), right click the website, save as `get-pip.py`, and run the code.

Now, you are all set !!!

### Dataset
We have 10 datasets of waypoints in town03 saved in `refmaps` folder. Also, we saved starting points of every refmap in `starting_points.csv`. You should not modify any of these.

### Training a model
To train a model, you just simply run `train.py`. The hyperparameters can be changed in `settings.py`. 
*Note that for 30,000 episodes, it takes about 9 days to finish.* So, make sure that your computer is ready.

While the model is training, you will notice `DATA` folder that keeps updating. This will save the progress of the training.

### Testing a model
When the model finished its training, you can test the model by running `run.py`. 
**But, before you run, make sure that the name of the model you want to test in `DATA` folder is the same as the one in `settings.py` in the `Test settings` section.**
After the testing is finished, the `Test results` folder will appear. That is where we keep the test results.

### Adjusting hyperparameters
You can change all hyperparameters in `settings.py`.

### Upgrading our code
If you want to improve our work, there is `sources` folder which contains `agent.py` - AGENT, `carla.py` - ENVIRONMENT, `models.py` - ACTOR & CRITIC. This is where all the fun things started. 

Furthermore, there is `replayBuffer.py` for `agent.py` but I don't recommend changing this file, only if you know what you are doing.

### Final note
1. I uploaded our best model (Model 5) in `DATA` folder and its test result in `Test results` folder. If you want to see how the car we developed drives, you can run our model in `run.py`.
2. There is a [demo video](https://youtu.be/hJAFLDik3_c) here explained in Thai. 

## Hope you have fun with our project !!!
Last updated: May 19th, 2022 18:52
