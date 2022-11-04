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
import math

###############
# Naive Bayes #
###############
class NaiveBayes:
    def __init__(self, bayesConfig, n, test, common, queries=None ):

        # Configuration
        self.bayesConfig = bayesConfig
        self.number = n
        self.fileName = self.bayesConfig["learnfile" + str(n)]
        self.given = self.bayesConfig["given" + str(n)]
        self.test = test
        self.outFile = self.bayesConfig["out" + str(n)]
        self.commonSettings = common
        self.queries = queries
        self.structure = self.bayesConfig["structure" + str(n)]
        self.listVars = []

        try:
            self.dp = int(self.commonSettings['decimalplaces'])
        except:
            self.dp = 3

        # Total for 'given' dependent column
        self.total = 0
        
        # Dictionary to hold discrete and learnt probabilities
        self.discretes = {}
        self.learnt = {}
        self.df = None
        self.variables = []
        self.questions = []
        
        # Do functions for learn or test mode
        if not self.test:
            self.makeStructure()
            self.countRows()
            self.getDiscreteVariables()
            self.getDiscreteValues()
            self.learn()
            self.saveLearnt()
            self.saveLearntText()
            self.displayLearnt()
            #self.displayDiscretes()
            
        elif self.test:
            self.makeStructure()
            self.loadLearnt()
            self.readQuestions()
            self.answerQuestion()

    ##################
    # parseStructure #
    ##################
    def parseStructure(self):
        s = self.structure
        s = s.replace('|', ',')
        s = s.replace('P(', '')
        s = s.replace(')', '')
        self.listVars = s.split(',')


    #################
    # makeStructure #
    #################
    def makeStructure(self):
        '''
        Reads in train or test file and removes 
        variables that are not in the structure.
        '''
        self.df = pd.read_csv(self.fileName)
        self.parseStructure()
        self.df = self.df[self.listVars]

    ########
    # show #
    ########
    def show(self, line = '', endLine = '\n'):
        if self.commonSettings['display'] == 'True':
            print(str(line), end=endLine)
        if self.commonSettings['logging'] == 'True':
            with open(self.commonSettings['logfile'], 'a') as f:
                f.write(str(line))
                f.write(endLine)

    #################
    # readQuestions #
    #################
    def readQuestions(self):
        count = 0
        removedVars = []
        # Read test file
        with open(self.fileName, 'r') as f:
            for line in f:
                line = line.strip().split(',')
                
                if count == 0:
                    print(self.listVars)
                    for index,var in enumerate(line):
                        if var not in self.listVars:
                            removedVars.append(index)
                            
                    for i in removedVars:
                        line.pop(i)
                    self.variables = line    

                else:
                    for i in removedVars:
                        line.pop(i)
                    self.questions.append(line)
                count += 1
        print(removedVars)
        print(self.variables)


    ##################
    # answerQuestion #
    ##################
    def answerQuestion(self):
        count = 0
        correct = 0
        char = " * "
        if self.commonSettings['log'] == 'True':
            char = " + "

        # Each question needs answering - they are P queries
        for question in self.questions:
            q = self.given
            
            answers = {}
            for discrete in self.learnt.keys():
                if "P("+q+"=" in discrete:
                    answers[discrete] = [[discrete, self.learnt[discrete]]]
                    for index, option in enumerate(question):
                        if self.variables[index] != q:
                            answers[discrete].append(["P("+ self.variables[index] + "='" + option + "'|" + discrete[2:-1]+")", self.learnt["P("+ self.variables[index] + "='" + option + "'|" + discrete[2:-1]+")"]])
            
            # Build evidence strings for human output
            for key in answers.keys():
                self.show(key[0:-1] + "|evidence) = ", '')
                for index, p in enumerate(answers[key]):
                    self.show(p[0], '')
                    if index < len(answers[key]) - 1:
                        self.show(char, '')
                self.show('', '\n')
            self.show('', '\n')

            # Build evidence ouput showing joint probabilities
            results = {}

            for key in answers.keys():
                self.show(key[0:-1] + "|evidence) = ", '')
                if self.commonSettings['log'] == 'True':
                    probability = 0

                else: 
                    probability = 1
                for index, p in enumerate(answers[key]):
                    eachOne = p[1][1]
                    if self.commonSettings['log'] == 'True':
                        eachOne = math.log(eachOne)
                        probability +=  eachOne
                    else:
                        probability *=  eachOne
                    self.show(round(eachOne, self.dp), '')
                    if index < len(answers[key]) - 1:
                        self.show(char, '')
                self.show(" = " + str(round(probability, self.dp)))
                results[key[0:-1]+"|evidence)"] = [probability]

            # Construct result
            for key in results.keys():
                for otherKey in results.keys():
                    if otherKey != key:
                        results[key].append(results[otherKey][0])

            self.show()

            # Output normalised result
            for key in results.keys():
                if self.commonSettings['log'] == 'True':
                    results[key][0] = math.exp(results[key][0])
                    results[key][1] = math.exp(results[key][1])
                numerator = results[key][0]
                denominator = sum(results[key])
                results[key].append(numerator / denominator)
                self.show(key + " = "  + str(round(results[key][-1],self.dp)))

            # Get prediction, argmax
            prediction = None
            value = 0
            for key in results.keys():
                if results[key][2] > value:
                    value = results[key][2]
                    # Pull argmax value from within human readable form
                    prediction = key.split('|')[0].replace('P('+q+'=','').replace("'",'')

            # Make metric - will be accuracy        
            count += 1
            if prediction ==  question[-1]:
                correct += 1
                self.show("Correct")
            else:
                self.show("Wrong")            
            self.show('')

        self.show("Correct predictions  : " + str(correct))
        self.show("Number of predictions: " + str(count))
        self.show("Accuracy = " + str(round(correct / count, self.dp)))
        

    ##############
    # loadLearnt #
    ##############
    def loadLearnt(self):
        with open(self.outFile, 'r') as f:
            self.learnt = json.load(f)

    ##############
    # saveLearnt #
    ##############
    def saveLearnt(self):
        with open(self.outFile, 'w') as f:
            json.dump(self.learnt, f)

    ##################
    # saveLearntText #
    ################## 
    def saveLearntText(self):
        with open(self.outFile + ".txt", 'w') as fp:
            for item in self.learnt.items():
                fp.write(item[0] + " = " + item[1][0] + " = " + str(round(item[1][1], self.dp)) +"\n")

    #################
    # displayLearnt #
    #################
    def displayLearnt(self):
        for item in self.learnt.items():
            self.show(item[0] + "=" + item[1][0] + "=" + str(round(item[1][1], self.dp)))


    #########
    # learn # 
    #########
    def learn(self):
        for variable in self.discretes:
            for attribute in self.discretes[variable]:
                if variable != self.given:
                    for givenOption in self.discretes[self.given]:
                        result = self.df[[variable,self.given]]
                        givenCount = result[result[self.given] == givenOption].shape[0]
                        # P(feature|given) = fraction = probability
                        result2 = result[(result[variable] == attribute) & (result[self.given] == givenOption)]
                        result3 = result[(result[variable] == attribute)]
                        variableCount = result2.shape[0]
                        attributeCount = result3.shape[0]
                        # Store in dictionary
                        # There are 3 possible values when variable count = 0
                        # No laplacian (hope there are no zeros)
                        # Simple Laplacian - add a small value to avoid zero
                        # Laplacian Smoothing - better
                        if variableCount == 0:
                            if self.commonSettings['laplaciansimple'] == 'True':
                                self.learnt["P(" + str(variable) + "='" + str(attribute) + "'|" + str(self.given) + "='" + str(givenOption) + "')" ] = (str(variableCount) + "/" + str(givenCount), float((variableCount + float(self.commonSettings['simple']))/ givenCount))
                            elif self.commonSettings['laplaciansimple'] == 'False':
                                if self.commonSettings['laplaciansmoothe'] == 'False':
                                    self.learnt["P(" + str(variable) + "='" + str(attribute) + "'|" + str(self.given) + "='" + str(givenOption) + "')" ] = (str(variableCount) + "/" + str(givenCount), float(variableCount / givenCount))
                                elif self.commonSettings['laplaciansmoothe'] == 'True':
                                    self.learnt["P(" + str(variable) + "='" + str(attribute) + "'|" + str(self.given) + "='" + str(givenOption) + "')" ] = (str(variableCount) + "/" + str(givenCount), float((variableCount + float(self.commonSettings['laplacian'])) / (givenCount + (float(self.commonSettings['laplacian']) * attributeCount))))
                        else:
                            # Two options here, Laplacian smoothing or not
                            # Because there are no zeros we are safe to not use Laplacian
                            # ...but we can use Laplacian
                            if self.commonSettings['laplaciansmoothe'] == 'False':
                                self.learnt["P(" + str(variable) + "='" + str(attribute) + "'|" + str(self.given) + "='" + str(givenOption) + "')" ] = (str(variableCount) + "/" + str(givenCount), float(variableCount / givenCount))
                            else:
                                self.learnt["P(" + str(variable) + "='" + str(attribute) + "'|" + str(self.given) + "='" + str(givenOption) + "')" ] = (str(variableCount) + "/" + str(givenCount), float((variableCount + float(self.commonSettings['laplacian'])) / (givenCount + (float(self.commonSettings['laplacian']) * attributeCount))))

    ####################
    # displayDiscretes #
    ####################
    def displayDiscretes(self):
        self.show("Discrete variables and probabilities for: " + str(self.fileName))
        self.show("Total samples:" + str(self.total))
        self.show()
        for variable in self.discretes:
            self.show("Var:" + str(variable))
            for option in self.discretes[variable]:
                self.show("\tAttribute: " + str(option))
                self.show("\tTotal: " + str(self.discretes[variable][option]['total']))
                self.show("\tProb : " + str(round(self.discretes[variable][option]['prob'], self.dp)))
                self.show()

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
                    self.discretes[variable][value[1]] = {'total': 1, 'prob': float(1/self.total)}
                    self.learnt["P(" + str(variable) + "='" + str(value[1]) + "')"] = ("1/"+str(self.total), float(1/self.total))
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

###############
# parseConfig #
###############
def parseConfig(config="Config.cfg"):
    '''
    Parses the config file and enters details into dictionaries.
    The format is detailed within the config file.
    '''
    common = {}
    bayesConfig = {}
    commonFlag = False
    fileNumber = 0
    queryNumber = 0
    queries = {}
    with open(config, 'r') as f:
        for line in f:
            # Comments and blank lines are ignored
            if line[0] == '#' or line[0] == '\n': continue
            line = line.strip().split(':')

            # Switch into common mode - these variables are for all models
            if line[0].lower() == "common":
                commonFlag = True
                continue

            # Switch into file mode. The variables control bayes reading and writing
            if line[0].lower() == "file":
                fileNumber += 1
                commonFlag = False
                continue 

            # Store
            if commonFlag:
                common[line[0].lower()] = line[1].strip()
            else:
                if "query" in line[0].lower():
                    queryNumber += 1
                    queries[line[0].lower()+str(fileNumber) + "-" + str(queryNumber)] = line[1].strip()
                else:
                    bayesConfig[line[0].lower()+str(fileNumber)] = line[1].strip()

    return common, bayesConfig, queries




########
# main #
########
def main(argv):
    common, bayesConfig, queries = parseConfig()
    
    open(common['logfile'], 'w').close()
    
    for n in range(1, 2):
        try:
            # Queries for this network
            q = [queries[key] for key in queries.keys() if "query"+str(n) in key]
            # Learn and save results
            NB = NaiveBayes(bayesConfig, n, False, common)
            # Test and save results
            NB = NaiveBayes(bayesConfig, n, True, common, q)
        except KeyError as e:
            print()
            print("All tests have been run. Please see results folder.", e)
            quit()
        else:
            print(common)
            print()
            print(bayesConfig)
            print()
            print(queries)



##################
# start properly #
##################
if __name__ == "__main__":

   main(sys.argv[1:])