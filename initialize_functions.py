# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 15:15:52 2018

@author: Miks
"""

import numpy as np
import pandas as pd
import random
def create_food_nutrients(foodnut, withValues=False):
    for nutrient in foodnut.columns.values:
        if np.sum(foodnut[nutrient].values)==0:
            foodnut.drop(columns=nutrient, inplace=True)
    new=foodnut.columns.values
    new[0]='Food'
    foodnut.columns=new
    foodnut.set_index('Food', inplace=True)
    foods=foodnut.index.values
    new=foodnut.to_dict(orient='index')
    for food in foods:
        for nutrient in foodnut.columns.values:
                if new[food][nutrient]==0:
                    del(new[food][nutrient])
        if withValues==False:
            new[food]=list(new[food].keys())
    return new

def default_preference(foodnut):
    
    prefs={}
    newfoods=random.sample(list(foodnut.index.values), 50) #random initial preference list
    for food in foodnut.index.values:
        prefs[food]=float(random.randint(0, 1))
    preferences=pd.Series(list(prefs.values()), index=foodnut.index.values)
    prefos=pd.concat([foodnut, preferences], axis=1)  
    columns=prefos.columns.values  
    columns[-1]='Preference'
    prefos.columns=columns   
    #prefos is all food sorted by default preference profile
    prefos.sort_values(by='Preference', ascending=False, inplace=True)
    return prefos

def init_learn_set(foodnut, prefos):
    newfoods=random.sample(list(foodnut.index.values), 50) #random initial preference list
    initlearn=prefos.loc[newfoods]
    return initlearn
