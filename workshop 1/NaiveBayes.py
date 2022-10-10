######################################################
# Module        :   Advanced Artificial Inteligence  #
# Workshop      :   1 - Naive Bayes                  #
# Author        :   Andy Perrett (PER18684092)       #
# Date          :   9th October 2022                 #
######################################################

# Standard imports
import sys

# Other imports
import pandas as pd
import json

###############
# Naive Bayes #
###############
class NaiveBayes:
    def __init__(self, fileName='play_tennis-train.csv', given='PT', test="False"):
        self.fileName = fileName
        self.given = given
        self.test = test

        # Total for 'given' dependent column
        self.total = 0
        
        # Dictionary to hold discrete and learnt probabilities
        self.discretes = {}
        self.learnt = {}
        self.df = None
 
        # Do functions for learn or test mode
        if test == "False":
            self.df = pd.read_csv(fileName)
            self.countRows()
            self.getDiscreteVariables()
            self.getDiscreteValues()
            self.learn()
            self.saveLearnt()
            self.saveLearntText()
        elif test == "True":
            self.loadLearnt()

    ##############
    # loadLearnt #
    ##############
    def loadLearnt(self):
        with open(self.fileName + ".probs", 'r') as f:
            self.learnt = json.load(f)

    ##############
    # saveLearnt #
    ##############
    def saveLearnt(self):
        with open(self.fileName + ".probs", 'w') as f:
            json.dump(self.learnt, f)

    ##################
    # saveLearntText #
    ################## 
    def saveLearntText(self):
        with open(self.fileName + ".probs.txt", 'w') as fp:
            for item in self.learnt.items():
                fp.write(item[0] + " = " + item[1][0] + " = " + str(round(item[1][1],3)) +"\n")

    #################
    # displayLearnt #
    #################
    def displayLearnt(self):
        for item in self.learnt.items():
            print(item[0],"=",item[1][0],"=",round(item[1][1],3))

    #########
    # learn # 
    #########
    def learn(self):
        for variable in self.discretes:
            for option in self.discretes[variable]:
                if variable != self.given:
                    for givenOption in self.discretes[self.given]:
                        result = self.df[[variable,self.given]]
                        givenCount = result[result[self.given] == givenOption].shape[0]
                        # P(feature|given) = fraction = probability
                        result2 = result[(result[variable] == option) & (result[self.given] == givenOption)]
                        variableCount = result2.shape[0]
                        # Store it dictionary
                        self.learnt["P(" + str(variable) +"='" + str(option) + "'|" + str(self.given) + "='" + str(givenOption) + "')" ] = (str(variableCount) + "/" + str(givenCount), float(variableCount / givenCount))

    ####################
    # displayDiscretes #
    ####################
    def displayDiscretes(self):
        print("Discrete variables and probabilities for: ", self.fileName)
        print("Total samples:", self.total)
        print()
        for variable in self.discretes:
            print("Var:", variable)
            for option in self.discretes[variable]:
                print("\tOption: ", option)
                print("\tTotal: ",self.discretes[variable][option]['total'])
                print("\tProb : ",round(self.discretes[variable][option]['prob'], 4))
                print()

    #####################
    # getDiscreteValues #
    #####################
    def getDiscreteValues(self):
        for variable in self.discretes:
            result = self.df[[variable]]
            result = result.reset_index()
            # Count each feature and add a discrete probability
            for index, value in result.iterrows():
                if value[1] not in self.discretes[variable]:
                    self.discretes[variable][value[1]] = {'total': 1, 'prob': float(0.0)}
                    self.learnt["P(" + str(variable) + "='" + str(value[1]) + "')"] = ("0/0", float(0.0))
                else:
                    self.discretes[variable][value[1]]['total'] += 1
                    self.discretes[variable][value[1]]['prob'] = self.discretes[variable][value[1]]['total'] / self.total
                    self.learnt["P("+str(variable)+"='"+str(value[1])+"')"] = (str(self.discretes[variable][value[1]]['total']) + "/"+ str(self.total)),float(self.discretes[variable][value[1]]['total'] / self.total)
                    
    ################################
    # populate discrete dictionary #
    ################################
    def getDiscreteVariables(self):
        for variable in self.df.columns:
            if variable not in self.discretes:
                self.discretes[variable] = {}
        
    ######################
    # get number of rows #
    ######################
    def countRows(self):
        self.total = self.df.shape[0]

########
# main #
########
def main(argv):

    # TODO redo cmd line processing
    try:
        if len(argv) == 0:
            NB = NaiveBayes()
        if len(argv) == 1:
            NB = NaiveBayes(argv[0]) 
        if len(argv) == 2:
            NB = NaiveBayes(argv[0], argv[1])
        if len(argv) == 3:
            NB = NaiveBayes(argv[0], argv[1], argv[2])

        NB.displayLearnt()

    except IndexError:
        print()
        print("Command line arguments are wrong")
        print()
        print("Usage:")
        print("------")
        print("/usr/bin/python NaiveBayes 'filename' 'given'")
        print("\twhere 'given' is the feature name as part of P(condition|given)")
        print()

##################
# start properly #
##################
if __name__ == "__main__":
   main(sys.argv[1:])