######################################################
# Module        :   Advanced Artificial Inteligence  #
# Workshop      :   Assignment 1                     #
# Author        :   Andy Perrett (PER18684092)       #
# Date          :   18th October 2022                #
######################################################

# Main imports
import json
import math
import sys
import pandas as pd
from sklearn.metrics import confusion_matrix

# My imports
from Combinations import Combinations

###############
# Naive Bayes #
###############
class NaiveBayes:
    def __init__(self, bayesConfig, n, test, common, queries=None ):

        # Config and general control
        self.number = n
        self.test = test

        # Configuration
        self.bayesConfig = bayesConfig
        self.fileName = self.bayesConfig["learnfile" + str(n)]
        self.fileNameTest = self.bayesConfig["testfile" + str(n)]    
        self.outFile = self.bayesConfig["out" + str(n)]
        self.commonSettings = common
        self.queries = queries
        self.structure = self.bayesConfig["structure" + str(n)]
        self.findBestStrure = self.bayesConfig["findbeststructure" + str(n)]
        self.MLE = self.commonSettings['maximumlikelihood']
        self.lapSimple = self.commonSettings['laplaciansimple']
        
        try:
            self.dp = int(self.commonSettings['decimalplaces'])
        except:
            self.dp = 3

        # Preparation
        self.listVars = self.parseStructure(self.structure)
        
        self.given = self.listVars[0]

        # Total for 'given' dependent column
        self.total = 0
        self.bestStructure = []
        self.bestAcc = 0
        self.numberStructures = 0
        self.overrideDisplay = False
        # Dictionary to hold discrete learnt probabilities
        self.discretes = {}
        self.learnt = {}
        self.df = None
        self.variables = []
        self.questions = []

        self.rawDataDict = {}
        self.randVariables = []
        
        # Do functions for learn or test mode
        if not self.test:
            self.readFile(self.fileName)
            #self.makeStructure(self.fileName)
            self.countRows()
            self.getDiscreteVariables()
            self.getDiscretePriors()
            self.learn()
            self.saveLearnt()
            self.saveLearntText()
            self.displayLearnt()          
        elif self.test:
            combos = self.createStructures()
            combos.insert(0,self.listVars)
            self.numberStructures = len(combos)
            for i, v in enumerate(combos):
                # Don't want to log/display while testing
                if i > 0:
                    self.overrideDisplay = True
                else: 
                    self.overrideDisplay = False
                self.listVars = v
                #self.makeStructure(self.fileNameTest)
                #self.readFile(self.fileNameTest)
                self.loadLearnt()
                self.readQuestions()
                self.answerQuestions()

    ############
    # readFile #
    ############
    def readFile(self, fileName):
        rawData = []       
        with open(fileName, 'r') as f:
            count = 0
            for line in f:
                if count == 0:
                    self.randVariables = line.strip().split(',')
                    self.variables = self.randVariables
                else:
                    data = line.strip().split(',')
                    rawData.append(data)
                count += 1
        for index, variable in enumerate(self.randVariables):
            self.rawDataDict[variable] = []
            for line in rawData:
                self.rawDataDict[variable].append(line[index])


    ####################
    # createStructures #
    ####################
    def createStructures(self):
        combos = []
        if self.findBestStrure:
            for n in range(1,len(self.listVars)-1):
                evidence = []
                for a in self.listVars:
                    if a != self.given:
                        evidence.append(a)
                combo = Combinations(evidence, len(evidence)-n)
                c = combo.getCombinations()

                # Insert target variable into each combo
                
                for index, variables in enumerate(c):
                    c[index].insert(0, self.given)
                    combos.append(c[index])
        return combos

    ##################
    # parseStructure #
    ##################
    def parseStructure(self, structure):
        '''
        The structure is given in the config file.
        A structure is P(target|evidence) where
        evidence is a list of variables.
        '''
        s = structure
        s = s.replace('|', ',')
        s = s.replace('P(', '')
        s = s.replace(')', '')
        return s.split(',')

    #################
    # makeStructure #
    #################
    def makeStructure(self, fileName):
        '''
        Reads in train or test file and removes 
        variables that are not in the structure.
        '''
        self.df = pd.read_csv(fileName)
        self.df = self.df[self.listVars]

    ########
    # show #
    ########
    def show(self, line = '', endLine = '\n'):
        '''
        Either prints to screen, writes to a log file
        or both.
        '''
        if self.overrideDisplay: return

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
        '''
        Questions / queries are actually the test file
        rows where predictions are made. Each question
        is asking given the evidence can we predict and
        outcome.
        '''
        self.questions = [] 
        self.variables = []  
          
        with open(self.fileNameTest, 'r') as f:
            count = 0
            for line in f:
                if count == 0:
                    vars = line.strip().split(',')
                    indexes = []
                    for index, variable in enumerate(vars):
                        if variable in self.listVars:
                            indexes.append(index)
                            self.variables.append(variable)
                else:
                    data = line.strip().split(',')
                    for index in indexes:
                        self.questions.append(data[index])
                count += 1
            print(self.questions)
            print(self.variables)


    ###########  
    # getQPos #
    ###########  
    def getQPos(self):
        '''
        Q / Question position is where in the list of
        variables is the target / the variable that we
        are trying to predict. The position could change.
        '''
        for i, v in enumerate(self.variables):
            if v == self.given:
                return i

    def validVariable(self, discrete):
        d = discrete[2:-1].split('|')
        d = d[0].split('=')
        return d[0]


    ##############
    # getAnswers #
    ##############
    def getAnswers(self, question):
        '''
        Answers are in the form:
        P(target='0'|evidence) = P(target='0') + P(age='3'|target='0') + P(oldpeak='0'|target='0')
        P(target='1'|evidence) = P(target='1') + P(age='3'|target='1') + P(oldpeak='0'|target='1')
        they are in human readable format.
        This function prepares the answers.
        '''
        answers = {}
        print("q",question, self.questions)
        for discrete in self.learnt.keys():
            if "P(" + self.given + "=" in discrete:
                answers[discrete] = [[discrete, self.learnt[discrete]]]
                print("ad", answers[discrete])
                try:
                    for index, option in enumerate(question):
                        if self.variables[index] != self.given:
                            answers[discrete].append(["P("+ self.variables[index] + "='" + option + "'|" + discrete[2:-1]+")", self.learnt["P("+ self.variables[index] + "='" + option + "'|" + discrete[2:-1]+")"]])
                except:
                    del answers[discrete]
        return answers

    ###################
    # displayEvidence #
    ###################
    def displayAnswers(self, answers, char):
        '''
        Displays such as:
        P(target='0'|evidence) = P(target='0') + P(age='3'|target='0') + P(oldpeak='0'|target='0')
        '''
        # Build evidence strings for human output
        for key in answers.keys():
            self.show(key[0:-1] + "|evidence) = ", '')
            for index, p in enumerate(answers[key]):
                self.show(p[0], '')
                if index < len(answers[key]) - 1:
                    self.show(char, '')
            self.show()
        self.show()

    ####################
    # enumerateAnswers #
    ####################
    def enumerateAnswers(self, answers, char):
        '''
        Enumerates the prepared human readable answers into calculations.
        P(target='0'|evidence) = -0.709 + -1.314 + -2.19 + -0.968 = xyz
        P(target='1'|evidence) = -0.677 + -1.937 + -1.301 + -2.099 = xyz
        '''
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
        return results    

    ###################
    # constructResult #
    ###################
    def constructResult(self, results):
        '''
        The normalisation requires the previous two results to use
        each others value within the denominator.
        This is a helper function.
        '''
        for key in results.keys():
            for otherKey in results.keys():
                if otherKey != key:
                    results[key].append(results[otherKey][0])
        return results

    ####################
    # normaliseResults #
    ####################
    def normaliseResults(self, results):
        '''
        Does what it says on the tin. Normalises the two answers and
        calls each value a result.
        '''
        # Output normalised result
        self.show()
        for key in results.keys():
            if self.commonSettings['log'] == 'True':
                results[key][0] = math.exp(results[key][0])
                results[key][1] = math.exp(results[key][1])
            numerator = results[key][0]
            denominator = sum(results[key])
            results[key].append(numerator / denominator)
            self.show(key + " = "  + str(round(results[key][-1],self.dp)))
        return results

    ####################
    # argMaxPrediction #
    ####################
    def argMaxPrediction(self, results):
        '''
        After normalising, which is the max value as that is
        the prediction
        '''
        prediction = None
        value = 0
        for key in results.keys():
            if results[key][2] > value:
                value = results[key][2]
                # Pull argmax value from within human readable answer
                prediction = key.split('|')[0].replace('P(' + self.given + '=','').replace("'",'')
        return prediction

    ##################
    # answerQuestion #
    ##################
    def answerQuestions(self):
        '''
        Main function that takes each line of a test file
        (which I call questions / queries) and produces an
        answer for each row. Discrete values have been
        previously learnt.
        '''
        # For basic metrics
        # NOTE extend so using ones mentioned in ASS Brief
        count = 0
        correct = 0
        y = []
        yHat = []

        # Get position of the target variable
        qPosition = self.getQPos()
        # Log and standard have different math operations
        char = " * "    
        if self.commonSettings['log'] == 'True':
            char = " + "

        # Each question needs answering - they are P queries
        # found in each row of test file
        self.show()
        self.show(self.fileNameTest)
        self.show("Test Results")
        self.show("------------")
        self.show()
        for question in self.questions:
            answers = self.getAnswers(question)
            self.displayAnswers(answers, char)
            results = self.enumerateAnswers(answers, char)
            results = self.constructResult(results)
            results = self.normaliseResults(results)
            prediction = self.argMaxPrediction(results)
            
            # Make metrics        
            count += 1
            print(question, qPosition)
            y.append(question[qPosition])
            yHat.append(prediction)
            if prediction ==  question[qPosition]:
                correct += 1
                self.show("Correct")
            else:
                self.show("Wrong")            
            self.show()

        self.show(self.listVars)
        self.show("Correct predictions  : " + str(correct))
        self.show("Number of predictions: " + str(count))
        self.show("Accuracy = " + str(round(correct / count, self.dp)))
        self.show()

        # print("Structure", self.listVars)
        # print("Correct predictions  : " + str(correct))
        # print("Number of predictions: " + str(count))
        # print("Accuracy = " + str(round(correct / count, self.dp)))
        # print()

        if correct / count > self.bestAcc:
            self.bestAcc = correct / count
            self.bestStructure = self.listVars
            print("Best Structure: " +str(self.bestStructure) + " Acc: " + str(round(self.bestAcc * 100.0, self.dp)) + "% Combos tried: " + str(self.numberStructures))

        # Confusion matrix
        try:
            tn, fp, fn, tp = confusion_matrix(y, yHat).ravel()
            self.show("tn, fp, fn, tp: " + str(tn) + " " + str(fp) + " " + str(fn) + " " + str(tp))
        except:
            # We are here is there is not enough tests as in the 
            # play_tennis example
            pass

    ##############
    # loadLearnt #
    ##############
    def loadLearnt(self):
        '''
        The model previously learnt discrete probabilities
        so they can be loaded back in as a dictionary.
        '''
        with open(self.outFile, 'r') as f:
            self.learnt = json.load(f)

    ##############
    # saveLearnt #
    ##############
    def saveLearnt(self):
        '''
        The classifier has learnt discrete probabilities.
        These are saved for future use as a dictionary.
        '''
        with open(self.outFile, 'w') as f:
            json.dump(self.learnt, f)

    ##################
    # saveLearntText #
    ################## 
    def saveLearntText(self):
        '''
        All the discrete / conditional probabilities learnt are saved
        in human readable format - just because we can.
        '''
        with open(self.outFile + ".txt", 'w') as fp:
            for item in self.learnt.items():
                fp.write(item[0] + " = " + item[1][0] + " = " + str(round(item[1][1], self.dp)) +"\n")

    #################
    # displayLearnt #
    #################
    def displayLearnt(self):
        '''
        All learning probabilities can be displayed or logged.
        '''
        self.show(self.fileName)
        self.show("Learnt Probabilities")
        self.show("--------------------")
        for item in self.learnt.items():
            self.show(item[0] + "=" + item[1][0] + "=" + str(round(item[1][1], self.dp)))

    #########
    # learn # 
    #########
    def learn(self):
        '''
        Learns conditional probabilities. P(variable=atrribute|evidence=attribute).
        None, Simple Laplacian or Maximum Likelihood is applied to avoid zeros.
        '''
        for variable in self.discretes:
            for attribute in self.discretes[variable]:
                if variable != self.given:
                    for givenOption in self.discretes[self.given]:
                        countXY = self.countMatchingAttribsMulti(variable, self.given, attribute, givenOption)
                        countY = self.countMatchingAttribs(self.given, givenOption)
                        domainSizeX = len(self.discretes[variable])
                        # There are 3 possible values when variable count = 0
                        if countXY == 0:
                            if self.MLE == 'True':
                                self.learnt["P(" + str(variable) + "='" + str(attribute) + "'|" + str(self.given) + "='" + str(givenOption) + "')" ] = (str(countXY + 1) + "/" + str(countY + domainSizeX), float((countXY + 1) / (countY + domainSizeX)))
                            elif self.lapSimple == 'True':
                                self.learnt["P(" + str(variable) + "='" + str(attribute) + "'|" + str(self.given) + "='" + str(givenOption) + "')" ] = (str(countXY) + "/" + str(countY), float((countXY + float(self.commonSettings['simple'])) / countY))
                            elif self.lapSimple == 'False':
                                    self.learnt["P(" + str(variable) + "='" + str(attribute) + "'|" + str(self.given) + "='" + str(givenOption) + "')" ] = (str(countXY) + "/" + str(countY), float(countXY / countY))
                        else:
                            if self.MLE == 'True':
                                self.learnt["P(" + str(variable) + "='" + str(attribute) + "'|" + str(self.given) + "='" + str(givenOption) + "')" ] = (str(countXY + 1) + "/" + str(countY + domainSizeX), float((countXY + 1) / (countY + domainSizeX)))
                            else:
                                self.learnt["P(" + str(variable) + "='" + str(attribute) + "'|" + str(self.given) + "='" + str(givenOption) + "')" ] = (str(countXY) + "/" + str(countY), float(countXY / countY))

    ########################
    # countMatchingAttribs #
    ########################    
    def countMatchingAttribs(self, variable, match):
        count = 0
        for attrib in self.rawDataDict[variable]:
            if attrib == match:
                count += 1
        return count

    #############################
    # countMatchingAttribsMulti #
    #############################
    def countMatchingAttribsMulti(self, variable1, variable2, match1, match2):
        count = 0
        for (attrib1, attrib2) in zip(self.rawDataDict[variable1], self.rawDataDict[variable2]):
            if attrib1 == match1 and attrib2 == match2:
                count += 1
        return count


    def getVariableContents(self,listOfVars):
        dict = {}
        for variable in listOfVars:
            dict[variable] = self.rawDataDict[variable]
        return dict

    ####################
    # displayDiscretes #
    ####################
    def displayDiscretes(self):
        '''
        What it says on the tin - used for debugging and testing.
        '''
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
    # getDiscretePriors #
    #####################
    def getDiscretePriors(self):
        '''
        Simple variable probabilities P(variable=attribute) are found.
        These must all sum to 1 across the variables attributes.
        '''
        for variable in self.discretes:
            result = self.rawDataDict[variable]
  
            # Count each feature and add a discrete probability
            for attribute in result:
                if attribute not in self.discretes[variable]:
                    self.discretes[variable][attribute] = {'total': 1, 'prob': float(1/self.total)}
                    domainSizeX = len(self.discretes[variable])
                    if self.MLE == 'True':
                        self.learnt["P(" + str(variable) + "='" + str(attribute) + "')"] = ("2/"+str(self.total + domainSizeX)), float(2/(self.total + domainSizeX))
                    elif self.lapSimple == 'True':
                        self.learnt["P(" + str(variable) + "='" + str(attribute) + "')"] = ("1/"+str(self.total)), float(1/self.total)
                else:
                    self.discretes[variable][attribute]['total'] += 1
                    countX = self.discretes[variable][attribute]['total']
                    domainSizeX = len(self.discretes[variable])
                    self.discretes[variable][attribute]['prob'] = (countX + 1) / (self.total + domainSizeX)
                    if self.MLE == 'True':
                        self.learnt["P("+str(variable)+"='"+str(attribute)+"')"] = (str(countX + 1) + "/"+ str(self.total + domainSizeX)),float((countX + 1) / (self.total + domainSizeX))

    ################################
    # populate discrete dictionary #
    ################################
    def getDiscreteVariables(self):
        '''
        Builds a dictionary of variables with discrete values.
        '''
        for variable in self.randVariables:
            if variable not in self.discretes:
                self.discretes[variable] = {}
        
    ######################
    # get number of rows #
    ######################
    def countRows(self):
        self.total = len(self.rawDataDict[self.randVariables[0]])

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
                # Each query and file name gets a number
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

    # Config is own format
    common, bayesConfig, queries = parseConfig()
    
    # Clear log file on run
    open(common['logfile'], 'w').close()
    
    # NOTE reduce 10 to number of files in config
    # needs to be automated 
    for n in range(1, 6):
        try:
            # Queries for this network
            q = [queries[key] for key in queries.keys() if "query"+str(n) in key]
        
            # Learn and save results
            NB = NaiveBayes(bayesConfig, n, False, common)
            # Test and save results
            NB = NaiveBayes(bayesConfig, n, True, common, q)
        
        # Only need except due to for loop
        except KeyError as e:
            print(e)
            #print("All tests have been run. Please see results folder.")
            #quit()
        else:
            #print(common)
            print()
            #print(bayesConfig)
            #print()
            #print(queries)

##################
# start properly #
##################
if __name__ == "__main__":
   main(sys.argv[1:])