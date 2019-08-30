import numpy as np
import random
from world_model import World


class correct:

    def __init__(self):
        self.mate_posx_data = np.zeros((15,11))
        self.mate_posy_data = np.zeros((15,11))
        self.opp_posx_data = np.zeros((15,11))
        self.opp_posy_data = np.zeros((15,11))

    def data_correct(self,string):
        wm = World(string)
        for i in range(0,15):
            cycle = random.randint(0,5999)
            if cycle == 3000:
                cycle -= 1
                
            for unum in range(0,11):
                self.mate_posx_data[i][unum] = wm.ourPlayer_x(unum+1,cycle)
                self.mate_posy_data[i][unum] = wm.ourPlayer_y(unum+1,cycle)
                self.opp_posx_data[i][unum] = wm.theirPlayer_x(unum+1,cycle)
                self.opp_posy_data[i][unum] = wm.theirPlayer_y(unum+1,cycle)
