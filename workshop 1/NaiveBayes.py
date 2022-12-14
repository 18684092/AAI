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
    def __init__(self, fileName='play_tennis-test.csv', given='PT', test="True"):
        self.fileName = fileName
        self.given = given
        self.test = test

        # Total for 'given' dependent column
        self.total = 0
        
        # Dictionary to hold discrete and learnt probabilities
        self.discretes = {}
        self.learnt = {}
        self.df = None
        self.variables = []
        self.questions = []

        # Do functions for learn or test mode
        if self.test == "False":
            self.df = pd.read_csv(fileName)
            self.countRows()
            self.getDiscreteVariables()
            self.getDiscreteValues()
            self.learn()
            self.saveLearnt()
            self.saveLearntText()
            self.displayLearnt()
        elif self.test == "True":
            self.loadLearnt()
            self.readQuestions()
            self.answerQuestion()

    #################
    # readQuestions #
    #################
    def readQuestions(self):
        count = 0
        with open(self.fileName, 'r') as f:
            for line in f:
                line = line.strip().split(',')
                if count == 0:
                    self.variables = line
                else:
                    self.questions.append(line)
                count += 1

    ##################
    # answerQuestion #
    ##################
    def answerQuestion(self):
        for question in self.questions:
            q = ''
            for index, option in enumerate(question):
                if option == '?':
                     q = self.variables[index] 
                     break
            # we have question q
            answers = {}
            for discrete in self.learnt.keys():
                if "P("+q+"=" in discrete:
                    answers[discrete] = [[discrete, self.learnt[discrete]]]
                    for index, option in enumerate(question):
                        if option != '?':
                            answers[discrete].append(["P("+ self.variables[index] + "='" + option + "'|" + discrete[2:-1]+")", self.learnt["P("+ self.variables[index] + "='" + option + "'|" + discrete[2:-1]+")"]])
            
            # Build evidence strings for human output
            for key in answers.keys():
                print(key[0:-1]+"|evidence) = ", end='')
                for index, p in enumerate(answers[key]):
                    print(p[0], end='')
                    if index < len(answers[key]) - 1:
                        print(" * ", end='')
                print()
            print()

            # Build evidence ouput showing joint probabilities
            results = {}
            for key in answers.keys():
                print(key[0:-1]+"|evidence) = ", end='')
                probability = 1
                for index, p in enumerate(answers[key]):
                    probability *= p[1][1] 
                    print(round(p[1][1],3), end='')
                    if index < len(answers[key]) - 1:
                        print(" * ", end='')
                print(" = ", round(probability,3))
                results[key[0:-1]+"|evidence)"] = [probability]
            print()


            # Construct result
            for key in results.keys():
                for otherKey in results.keys():
                    if otherKey != key:
                        results[key].append(results[otherKey][0])

            # Output result
            for key in results.keys():
                numerator = results[key][0]
                denominator = sum(results[key])
                results[key].append(numerator / denominator)
                print(key + " = "  + str(round(results[key][-1],3)))
                


    ##############
    # loadLearnt #
    ##############
    def loadLearnt(self):
        with open("learnt", 'r') as f:
            self.learnt = json.load(f)

    ##############
    # saveLearnt #
    ##############
    def saveLearnt(self):
        with open("learnt", 'w') as f:
            json.dump(self.learnt, f)

    ##################
    # saveLearntText #
    ################## 
    def saveLearntText(self):
        with open("learnt.txt", 'w') as fp:
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