# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 19:34:52 2018

@author: Miks
"""

import pandas as pd
import numpy as np
import random 
from guizero import App, Text, Window, PushButton, Picture, TextBox, yesno, info, CheckBox, ListBox
path='./data/'
foodnut=pd.read_csv(path+'food_nutrient_dataset_v2.csv')
foodnut.set_index('Unnamed: 0', inplace=True)
foods = foodnut.index.values
choice = random.sample(list(foods), np.random.randint(3, 7))
lowbios=random.sample(list(foodnut.columns.values), 4)
class Screen:
    def __init__(self):
        self.app=App(height=350, width=500)
        self.body=Text(self.app, text = "What your body need:", color = "black")
        self.bx = []
        self.clickchoice = []
        self.goodf=[]
        self.badf=[]
    def add_nutrients(self, missingnutrients):
        self.listbox=ListBox(self.app, items = missingnutrients, height=3, width=30)
        self.lowbio=missingnutrients
    def choose_food(self, foods):
        if len(self.bx)>0:
            self.clickchoice=[b.value for b in self.bx]
            for b in self.bx:
                if b.value==1:
                    self.goodf.append(b.text)
                else:
                    self.badf.append(b.text)
                print(self.goodf)
                print(self.badf)
                print(self.clickchoice)
            for b in self.bx:
                b.destroy()
            self.bx = []
        for food in foods:
            self.bx.append(CheckBox(self.app, text = food))
    def select_food(self, foods):
        self.choose_food(foods)
        self.owb=PushButton(self.app, text="Next", command = self.choose_food, args = [self.foods])
    def show(self):
        self.app.display()
scr=Screen()
scr.add_nutrients(lowbios)
scr.choose_food(foods)
scr.select_food(foods)
scr.show()