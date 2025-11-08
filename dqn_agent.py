# dqn_agent.py

import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np
import random
import config

class DQNAgent:
