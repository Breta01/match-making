# Match Making Optimization
This repository impllements a match making algorithm for game League of Legends using neural network and optimization.

## Installation
Install python requirements (recommended python is **3.8** or higher).
```bash
pip install requirements.txt
```

In case you want to use the data collection algorithm, you need to obtain your Riot Games API token and place it into `env.py`. The file should looks something like:

```python
RIOT_GAME_API = "RGAPI-a..."
```

## Running the code
Code consists of 5 main files which depends on output of each other. The order is as follows:

1. [`match_data_loader.py`](./match_data_loader.py)
  - This file collects the match data
2. [`player_data_loader.py`](./player_data_loader.py)
  - This file collects and calculates statistics about players
3. [`prediction_model.py`](./prediction_model.py)
  - This file trains the prediction model 
4. [`optimization.py`](./optimization.py)
  - This defines the optimization model
5. [`simulator.py`](./simulator.py)
  - This file runs the simulation

You can ran each of them just like (just replacing the file name):
```bash
python prediction_mode.py 
```

### Quick start
You **don't** need to run the first two files for collecting data as these files aree already provided. So you can run the code using next three commands:
```bash
python prediction_model.py
python optimization.py # This one is mainly for debugging it doesn't output anything
python simulator.py
```

## Visualisation
Finally when the `simulator.py` finishes (or already while it is running), you can display the results using `visualisation.ipynb`. You just need to start jupyter notebook (lab) and open the file:
```bash
jupyter lab
# Then open the file in jupyter
```

_The simulation is currently set to only 1000 steps are more steps can take a lot of time to run_

# Authors
[Anas Abdelhakmi](https://www.linkedin.com/in/anas-abdelhakmi-a74331182/)  
[Břetislav Hájek](https://www.linkedin.com/in/b%C5%99etislav-h%C3%A1jek-75167111b/)
