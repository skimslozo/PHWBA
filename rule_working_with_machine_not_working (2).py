# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 10:18:55 2018

@author: Miks
"""

#==============================================================================
#Importing libraries
#==============================================================================
import numpy as np
import pandas as pd
from pyknow import *
from initialize_functions import create_food_nutrients, default_preference, init_learn_set
from guizero import App, Text, Window, PushButton, Picture, TextBox, yesno, info, CheckBox, ListBox
import random
import math
from ANN import PreferenceModelNetwork
#==============================================================================
#Fetching the data neeeded
#==============================================================================
path='./data/'
foodnut=pd.read_csv(path+'food_nutrient_dataset_v2.csv')
foodnut.set_index("Unnamed: 0")
#==============================================================================
#Creating the necessary dictionaries for running/testing
#==============================================================================
#Food-nutrients dictionary
food_nutrients=create_food_nutrients(foodnut)      
bioList = list(foodnut.columns.values)
#==============================================================================
#Creating default preferences for foods
#==============================================================================  
prefos=default_preference(foodnut)
initlearn=init_learn_set(foodnut, prefos)
app = App(height=350, width=500)
body = Text(app, text = "What your body needs:", color = "black")
#Randomly generate initial levels for given biomarkers (ideally we'd know a healthy person nutrient profile, which we could start with)
bioLevel = []
for i in bioList:
    bioLevel.append(random.randint(80,100)/100)

origLevel = bioLevel   
##==============================================================================
##Create the intial dictionaries with the user, nutrient & simulation parameters
##==============================================================================   
#

health ={'age' : 21, 'weight' : 84.0, 'height' : 178.0, 'lowBios' : ['Chromium (Cr)', 'Zinc (Zn)', 'Iodine (I)', 'Manganese (Mn)' ], 'status': 0}
#health ={'age' : 21, 'weight' : 84.0, 'height' : 178.0, 'lowBios' : [], 'preferences': prefos, 'status': 'UPDATED'}
biomarkers = {'bioLevel' : bioLevel, 'bioList' : bioList, 'dummy' : 0, 'foodNutr' : food_nutrients, 'didEat' : 0}
varFacts = {'timeStep': 1, 'lowerLim' : 0.25, 'timeElapsed' : 0, 'timeUpdate' : 0, 'fed' : 0}
suggestion = {'suggestion': []}
#health ={'age' : 21}
##health ={'age' : 21, 'weight' : 84.0, 'height' : 178.0, 'lowBios' : [], 'preferences': prefos, 'status': 'UPDATED'}
#biomarkers = {'a':0}
#varFacts = {'b':0}
#suggestion = {'c' : 0}
suggestion_engine=PreferenceModelNetwork(initlearn[foodnut.columns.values], list(initlearn['Preference']))
suggestion_engine.train()
#==============================================================================
#External functions
#============================================================================== 

def MachineInp(lowBios2, goodFood, badFood):
    print(["apple", "orange", "pineapple", "carrot"])
    #lowBios2 contains all nutrients that need to be satisfied
    #goodfood is food they said yes to, badfoood they said no to
    #need a list of food to be returned, the list is orderless- just contains food that cover all the nutrients in #lowBios2
    
def OptimalFood(incrBioDict, rejected):
    #incrBioDict.keys will be a list of biómarkers in descending order of urgency, i.e. feed the person as much as the first ones as possible
    #rejected is a list of rejected food items
    #we will only give the person one food item when they are hungry
    print("a food item")
    
def BioForFood(foodItem, foodNutrients):
    #returns a list of the biomarkers in foodItem
    foodNutrients[foodItem]
    pass                          

def ReduceFood(foodbase, preflist): #--> Within the rule, preflist should be the direct output of ANN, formatting is taken care of within this function
    #Reduces food to top start% so here is where our preference comes into it
    start=0.5
    prefs=pd.Series(preflist.flatten(), index=foodnut.index.values)
    cik=pd.concat([foodbase, prefs], axis=1)
    cik.sort_values([0], ascending=False)
    test=cik[0:int(start*len(cik))]
    not_covered=True
    count=0
    print('Start')
    while not_covered:
        for nutrient in foodnut.columns.values:
            if np.sum(test[nutrient].values)==0:
                count+=1
                print(count)
                test=prefbase[0:int(len(prefbase)*start)+count]
        else:
            not_covered=False
    return test

def RichFood(preef, neednutrs):
    #of the top start% of preferred food we then sort by nutrient levels
    return ((preef[neednutrs]!=0).sum(1)).sort_values(ascending=False)

def FindFood(foodlist, nutrients):
    pass

def SubmitFood(rejectedFood, Accepted, foodbase, ANN): 
    """
    rejectedFood - list of rejected foods by the user, output from GUI
    Accpted - list of accepted foods by the user, output from GUI
    foodbase - database of foods (in our case - foodnut)
    ANN - neural network instance (in our case - suggestion_engine)
    
    """
    #foodNutr is a dictioanry for food to nutrients
    food=rejectedFood+Accepted
    response=[0.0]*len(rejectedFood)+[1.0]*len(Accepted)
    ANN.train(X_new = foodbase.loc[food], y_new = response)

def bioDecay(age, weight, height, pastLevel, time, i):
    global origLevel
    term1 = np.log(0.5)/(3600*24)
    term2 = 1+abs(2-height/weight)
    lam = -(term1*term2) + age*pow(10,-7)
    #print(lam)
#        
    #pastTime = np.log(abs(pastLevel))/lam

#                              
    #curLevel = math.exp(lamb*(pastTime+timeStep))
    curLevel = origLevel[i]*math.exp(-lam*(time))
    #curLevel = pastLevel - (weight/height)*(1/3600)*0.8 #i.e. in an hour, the levels hould drop by about 0.8
    #curLevel = 0.1
    return curLevel
                              
    
    
    

def print_engine(e):
    print("\nEngine: " + str(e))
    print("    facts: " + str(e.facts))
    print("    rules: " + str(e.get_rules()))
    if len(e.agenda.activations)>0 :
        print("    activations: ("+str(len(e.agenda.activations))+") "+ str([a.__repr__() for a in e.agenda.activations]))
    else : 
        print("    activations: " + "NO_ACTIVATIONS")
##==============================================================================
##Test ---IMPORTANT!!!!---
##==============================================================================
#
##for i in range(100):
##    missingbios=random.sample(list(foodnut.columns.values), 30)
##    tmp=RichFood(prefos, missingbios)
##    print(tmp.index[1], tmp[0])
##"""--> We can always find a food which satisfies any amount of random nutrients picks from the essential nutrient pool"""
#==============================================================================
#Rule-base classes
#==============================================================================   

class Health(Fact):
    """has: age, weight, height, lowBios"""
    pass

class Biomarkers(Fact):
    """has: all biomarkers"""
    pass    

class Various(Fact):
    """has: time-step for analysis, lower limit"""
    pass
    
class Suggestion(Fact):
    """has: suggested foods"""
    pass
#Need to gave back to a list of 10 from prefos then next list of 5

class NutrientInfo:
    @Rule(Fact("Time"),
          salience = 22)
    
    def go(self):
        print("hey")
        
    @Rule(Fact("Time"),
          AS.bi << Biomarkers(bioList = MATCH.bioList, bioLevel = MATCH.bioLevel),
          #AS.bi << Biomarkers(bioLevel = MATCH.bioLevel),
          AS.ht << Health(age = MATCH.age, weight = MATCH.weight, height = MATCH.height, lowBios = MATCH.lowBios, status = MATCH.status),
          AS.vs << Various(timeStep = MATCH.dt, lowerLim = MATCH.lowLim, timeUpdate = MATCH.timeUpdate, timeElapsed = MATCH.timeElapsed),
          TEST(lambda timeUpdate : timeUpdate == 0),
          salience=10)
    
    #updates all biomarkers according to model and adds critcal biomarkers to lowBios
    def update(self, bi, ht, vs, bioList, bioLevel, lowLim, age, weight, height, dt, lowBios, timeUpdate, status, timeElapsed):
    #def update(self):
        print("update")
        bioDict2 = {}
        count = 0
        for nutr in bioList:
            bioDict2[nutr] = bioLevel[count]
            count += 1
        timeElapsed += dt
        lowBios2 = []
        
        for elm in lowBios:
            lowBios2.append(elm)
        
        count = 0
        for key in bioDict2:
            pastLevel = bioDict2[key]
            bioDict2[key] = bioDecay(age, weight, height, pastLevel, timeElapsed, count)
            if bioDict2[key] <= lowLim:
                lowBios2.append(key)
            count += 1
                
#        if lowBios2 != []:
#            self.modify(vr, firstFood = 1)
        bioLevel2 = []
        count = 0
        for nutr in bioList:
            bioLevel2.append(bioDict2[nutr])
            count+=1
            
            
        
        self.modify(bi, bioLevel = bioLevel2)
        self.modify(ht, lowBios = lowBios2, status=0)
        self.modify(vs, timeUpdate = 1)
        
        print("here I am")
        print("hey")
                
        
        
    #When lowBios is occupied, this rule gets a list of food covering all nutrients and generates
    #a list of 1's and 0's corresponding to each food item-0 for I won't eat it. For all 1's,
    #the nutrients in these foods is removed from lowBios and the new lowBios is sent to get a new list
    #of food. The response is also returned to the function so that the machine learning can know
    #which foods have aleady been eaten.
    @Rule(AS.ht << Health(lowBios = MATCH.lowBios),
          AS.bi << Biomarkers(bioList = MATCH.bioList, bioLevel = MATCH.bioLevel, foodNutr = MATCH.foodNutr),
          TEST(lambda lowBios : list(lowBios) != []), 
          salience = 9)
    
    def getFood(self, bi, ht, lowBios, bioList, bioLevel, foodNutr):
        print("getFood")
                
        bioDict2 = {}
        count = 0
        for nutr in bioList:
            bioDict2[nutr] = bioLevel[count]
            count += 1
        
        lowBios2 = []
        
        for elm in lowBios:
            lowBios2.append(elm)
        listbox = ListBox(app, items = lowBios2, height=3, width=30)
        
        goodFood = []
        badFood = []
        lastRec = []
        food = RichFood(ReduceFood(foodnut, suggestion_engine.predict(foodnut)), lowBios2) #needs to skip 0
        lastRec = food
        #Food should be a list of foods contianing one or more for each biomarker
        bx = []
        def choose_food(foods, bx):                                       
            if len(bx) > 0:
                for b in bx:
                    if b.value==1:
                        goodFood.append(b.value)
                        for nutrient in foodNutr[b.value]:
                            if nutrient in lowBios:
                                lowBios.remove(nutrient)
                                listbox.remove(nutrient)
                    else:
                        badFood.append(b.value)
                    b.destroy()
                bx = []
            for food in foods:
                bx.append(CheckBox(app, text = food))
        print(food)
        choose_food(yeew, bx)           
        owb = PushButton(app, text="Next", command = choose_food, args = [yeew, bx])
        app.display()
#            for item in food[0:4]:
#                response = random.randint(0,1)
#                if response == 1:
#                    goodFood.append(item)
#                    for elm in foodNutr[item]:
#                        if elm in lowBios2:
#                            lowBios2.remove(elm)
#                    if(lowBios2 == []):
#                        break
#                else:
#                    badFood.append(item)
        
        for item in goodFood:
            for elm in foodNutr[item]:
                bioDict2[elm] = 1
                
                
        bioLevel2 = []
        count = 0
        for nutr in bioList:
            bioLevel2.append(bioDict2[nutr])
            count+=1
                    
            
        self.modify(bi, bioLevel = bioLevel2)
            
        self.modify(ht, lowBios = lowBios2)
        
       
        
       
        SubmitFood(badFood, goodFood, foodnut, suggestion_engine)
        print("hey")
        #this will be called when the user says they are hungary, ordering the biomarkers from lowest to smallest
  
    @Rule(AS.bi << Biomarkers(bioList = MATCH.bioList, bioLevel = MATCH.bioLevel, didEat = MATCH.didEat), 
          AS.ht << Health(age = MATCH.age, weight = MATCH.weight, height = MATCH.height, lowBios = MATCH.lowBios, status = MATCH.status),
          AS.vs << Various(timeStep = MATCH.dt, lowerLim = MATCH.lowLim, timeUpdate = MATCH.timeUpdate, fed = MATCH.fed),
          TEST(lambda status : status == 0),
          salience=0)
    
    #updates all biomarkers according to model and adds critcal biomarkers to lowBios
    def FunctEnd(self, bi, ht, vs, bioList, bioLevel, lowLim, age, weight, height, dt, lowBios, timeUpdate, fed, didEat):
        global timeFact, hungryFact
        self.modify(vs, timeUpdate = 0, fed = 0)
        self.modify(ht, status = 1)
        self.retract(timeFact)
        if didEat == 1:
            self.retract(hungryFact)
            self.modify(bi, didEat = 0)
        
        
        print("update")
        
            
            
    @Rule(Fact("Hungry"),
          AS.bi << Biomarkers(bioList = MATCH.bioList, bioLevel = MATCH.bioLevel, foodNutr = MATCH.foodNutr, didEat = MATCH.didEat),
          AS.vr << Various(fed = MATCH.fed),
          AS.ht << Health(status = MATCH.status),
          TEST(lambda fed: fed == 0),
          salience = 11) # higher salience then update beacause they already want to eat-makes updating lowBios easier
    
    def SuggestFood(self, bioLevel, bioList, vr, bioDict, foodNutr, didEat):
        print("suggestFood")
        
        bioDict2 = {}
        count = 0
        for nutr in bioList:
            bioDict2[nutr] = bioLevel[count]
            count += 1
        sortedBios = []
        
        sorted_names = sorted(bioDict2, key=bioDict2.__getitem__)
        for k in sorted_names:
            sortedBios.append(k)
            

        rejected = []
        
        
        foodList = RichFood(ReduceFood(foodnut, suggestion_engine.predict(foodnut)), sortedBios[0:9])  #Get a list based off 10 most needed nutrients
        foodList = list(foodList.index.values)
        response = random.randint(0,1)
        goodFood = []
        for i in foodList:
            response = random.randint(0,1)
            if response == 1: #Eat food
                break
            else:
                rejected.append(i)
            
                
        if goodFood == []:   #Need better handler
            print("Doesn't want any of these food items")
            
 
            
        nutrs = foodNutr[goodFood[0]] 
        for elm in nutrs:
            bioDict2[elm] = 1
            
        bioLevel2 = []
        count = 0
        for nutr in bioList:
            bioLevel2.append(bioDict2[nutr])
            count+=1
            
        self.modify(bi, bioLevel = bioLevel2, didEat = 1)
        self.modify(vr, fed = 1)
        
        
        SubmitFood(rejected, [i])
        
        SubmitFood(rejected, goodFood, foodnut, suggestion_engine)
        


        
      
class Person(KnowledgeEngine, NutrientInfo):
    
    @DefFacts()
    def first(self):
        yield Health(**health)
        yield Biomarkers(**biomarkers)
        yield Various(**varFacts)
    


#For modifying variables in the engine
        
def get_return(self,key):
    return self.returnv.get(key)

def set_return(self,returnvaluedict):
    self.returnv = returnvaluedict
        
timeStep = 1
timeDuration = 24*60*60
person = Person()
        
person.reset()
#when do we actually need to run engine doe spritn engine run the engine?

#Little test:
#person.declare(Fact("Time"))
#person.run()

hungryFlg = 0
for i in range(0,timeDuration, timeStep):
    
    timeFact = person.declare(Fact("Time"))
    
    #person.modify(person.facts[3](timeUpdate = MATCH.timeUpdate), timeUpdate = 1) #if timeUpdate is 1, update time
    
    
    if i in [6*3600, 12*3600, 18*3600]:
        hungryFact = person.declare(Fact("Hungry"))
        hungryFlg = 1
        #person.modify(person.facts[3](fed = MATCH.fed), fed = 1) #if fed is 1, feed him
      
        
    person.run()
    
        
    if hungryFlg == 1:
        #person.retract(hungryFact)
        hungryFlg = 0  #Miks
        
    #person.retract(timeFact)
    print_engine(person)

"""To Do's: still have to implement the ANN within the rulebase itself (check if rules have been set up appropriately, call the suggestion_engine.predict, implement the rule which would expand the learning set of ANN & train it with the new learning set), input->output data should now be formatted appropriately
initlearn - dataframe with the initial learn values
suggestion_engine: machine-learning instance"""