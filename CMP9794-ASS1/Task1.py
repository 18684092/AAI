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
import time
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, roc_curve, auc, brier_score_loss, f1_score
import numpy as np
import random

# My imports
from Combinations import Combinations

# While testing structures some of the metrics warn (NaN)
import warnings
warnings.filterwarnings('ignore') 

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
        self.avoidZeros = self.commonSettings['avoidzeros']
        self.positivePred = self.bayesConfig["positiveprediction" + str(n)]
        self.KLConstant = float(self.commonSettings['klconstant'])
        self.pSampling = self.bayesConfig["priorsampling" + str(n)]
        self.pSampleCount = int(self.bayesConfig["priorsamples" + str(n)])
        self.gaussian = self.bayesConfig["gaussian" + str(n)]
        self.rejection = self.bayesConfig["rejection" + str(n)]

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
        self.bestResults = {}
        self.bestResults['BestAcc'] = 0
        self.bestResults['BestStructure'] = []
        self.numberStructures = 0
        self.overrideDisplay = False
        self.oldListVars = []
        
        # Dictionary to hold discrete learnt probabilities
        self.discretes = {}
        self.learnt = {}
        self.df = None
        self.variables = []
        self.questions = []

        # These are added due to replacing Pandas dataframes with my own code for holding data
        self.rawDataDict = {}
        self.randVariables = []
        self.averages = {}
        self.stds = {}
        
        # Do functions for learn or test mode
        if not self.test and self.gaussian == "False" and self.queries == None:
            # Delete results file
            open(self.outFile + "-results.txt", 'w').close()

            trainStart = time.time()
            self.showS("Training on " + self.fileName)
            self.showS("------------" + '-' * len(self.fileName))
            self.readFile(self.fileName)
            #self.makeStructure(self.fileName)
            self.countRows()
            self.getDiscreteVariables()
            self.getDiscretePriors()
            self.learn()
            self.saveLearnt()
            self.saveLearntText()
            self.displayLearnt() 
            trainEnd = time.time()  
            self.showS()
            self.showS("Training time: " + str(round(trainEnd - trainStart, self.dp)) + " seconds")
            self.showS()    
            if self.pSampling == "True":
                self.priorSampling()

        elif self.test  and self.queries == None and self.gaussian == "False":
            
            testStart = time.time()
            self.showS("Testing on " + self.fileNameTest)
            self.showS('-' * len("Testing on " + self.fileNameTest))
            self.oldListVars = self.listVars
            combos = self.createStructures()
            combos.insert(0,self.listVars)
            self.numberStructures = len(combos)
            self.readFile(self.fileName)

            self.loadLearnt()
            self.countRows()
            self.getDiscreteVariables()
            self.getDiscretePriors()
            for i, v in enumerate(combos):
                # Don't want to log/display while testing combinations
                if i > 0:
                    self.overrideDisplay = True
                else: 
                    self.overrideDisplay = False
                self.listVars = v
                self.readTestQueries()
                self.answerQueries(i)
            endTest = time.time()
            self.showS()
            self.showS("Testing time: " + str(round(endTest - testStart, self.dp)) + " seconds to find the best structure from " + str(len(combos)) + " structures")
            self.showS() 

            # Test queries taken from config
        elif self.test and self.queries != None and self.queries != [] and self.gaussian == "False":
            self.loadLearnt()
            self.readTestQueries()
            self.total = 1
            self.getDiscreteVariables()
            self.getDiscretePriors()

            # Standard query answering
            for query in self.queries:
                self.show("\nQuery: " + query)
                self.questions = []
                self.variables = []
                q = query.split('|')[1].replace(')','').split(',')
                a = []
                for attrib in q:
                    a.append(attrib.split("=")[1].replace("'",''))
                self.questions.append(a)
                for v in q:
                    self.variables.append(v.split('=')[0].replace("'",''))
                self.variables.append(self.given)
                self.answerQuery(0, query)

            if self.rejection == "True":
                # Prior query answering
                for query in self.queries:
                    self.show("\nUsing Rejection Sampling\nQuery: " + query)
                    self.questions = []
                    self.variables = []
                    q = query.split('|')[1].replace(')','').split(',')
                    a = []
                    for attrib in q:
                        a.append(attrib.split("=")[1].replace("'",''))
                    self.questions.append(a)
                    for v in q:
                        self.variables.append(v.split('=')[0].replace("'",''))
                    self.variables.append(self.given)
                    q = {}
                    for i,v in enumerate(self.variables):
                        if v != self.given:
                            q[v] = self.questions[0][i]
                    self.rejectionSampling(self.variables, query,q)
            
        elif self.gaussian == "True" and not self.test:
            # Gaussian
            # Learn Gaussian
            open(self.outFile + "-gaussian-results.txt", 'w').close()
            self.showG("Gaussian learning of means/std devs for " + str(self.fileName))
            start = time.time()
            self.readFile(self.fileName)
            for key in self.rawDataDict.keys():
                if key != self.given:
                    self.variableAvg(key, self.given)
                    self.variableSTD(key, self.given)
            end = time.time()
            self.showG("Gaussian learning time:" + str(end-start))
            self.showG()
            # Predict on test file
            self.gaussianProb()

    

    ############
    # gaussian #
    ############
    def gaussianProb(self):
        start = time.time()
        self.readTestQueries()
        acc = 0
        y = []
        yHat = []
        infTimes = [] 
        tru = []           

        for variables in self.questions:
            for index, value in enumerate(variables):
                if self.variables[index] == self.given:
                    tru.append(int(value))

        # For each row in test file
        for variables in self.questions:
            iStart = time.time()
            probT = []
            probF = []
            truth = 0

            # for each variable's value in each roow
            for index, value in enumerate(variables):
                if self.variables[index] != self.given:
                    # Grab precalculated means / std
                    vMeanT = self.averages[self.variables[index]]['avgT']
                    vMeanF = self.averages[self.variables[index]]['avgF']
                    vStdT = self.stds[self.variables[index]]['stdT']
                    vStdF = self.stds[self.variables[index]]['stdF']
                    # The value being calculated
                    x = value

                    # Probabilities
                    eT = -0.5 * ((float(x)-vMeanT) / vStdT) * ((float(x)-vMeanT) / vStdT)
                    eF = -0.5 * ((float(x)-vMeanF) / vStdF) * ((float(x)-vMeanF) / vStdF)
                    pT = (1 / (vStdT * math.sqrt(2 * math.pi))) * math.exp(eT)
                    pF = (1 / (vStdF * math.sqrt(2 * math.pi))) * math.exp(eF)
                    probT.append(pT)
                    probF.append(pF)
                elif self.variables[index] == self.given:
                    # the target truth
                    truth = int(value)


            iEnd = time.time()
            infTimes.append(iEnd-iStart)

            # Metrics
            T = sum(tru) / len(tru)
            F = 1 - T
            # This is for non log - just MLE
            for p in probT:
                T += p
            for p in probF:
                F += p

            # Normalise
            normT = T/(T+F)
            normF = F/(T+F)

            # How's the prediction looing
            predict = 0
            if normT > normF:
                predict = int(self.positivePred)
            answer = "wrong"
            if predict == truth:
                answer = "correct"
                acc +=1
                yHat.append(1)
            else:
                yHat.append(0)
            y.append(truth)
            self.showG(str(normT) + ", " + str(normF) + ", " + str(predict) + ", " + str(truth) + ", " + str(answer))
        self.showG()
        self.showG("STD Accuracy: " + str(acc/len(self.questions)))
        self.showG("Balanced Acc: " + str(balanced_accuracy_score(y, yHat)))
        (tn, fp, fn, tp) = confusion_matrix(y,yHat).ravel()
        self.showG("Confusion matrix: TN:" + str(tn) + " FP:" + str(fp) + " FN:" + str(fn) + " TP:" + str(tp))
        end = time.time()
        self.showG("Overall Test time: " + str(end-start))
        self.showG("Mean inference time: " + str(sum(infTimes)/ len(infTimes)))
        fpr, tpr, _ = roc_curve(y, yHat, pos_label=int(self.positivePred))
        a = auc(fpr, tpr)
        self.showG("AUC: " + str(a))
        self.showG("F1-score: " + str(f1_score(y, yHat, pos_label=int(self.positivePred))))


        pass

    ###############
    # variableAVG #
    ###############
    def variableAvg(self, variable, target):
        self.averages[variable] = {self.positivePred:0, '0':0, 'countT':0, 'countF':0, 'avgT':0, 'avgF':0}
        for value, y in zip(self.rawDataDict[variable], self.rawDataDict[target]):
            if y == self.positivePred:
                self.averages[variable][self.positivePred] += float(value)
                self.averages[variable]['countT'] += 1
                self.averages[variable]['avgT'] = self.averages[variable][self.positivePred] / self.averages[variable]['countT']
            else:
                self.averages[variable]['0'] += float(value)
                self.averages[variable]['countF'] += 1
                self.averages[variable]['avgF'] = self.averages[variable]['0'] / self.averages[variable]['countF']

    ###############
    # variableSTD #
    ###############
    def variableSTD(self, variable, target):
        self.stds[variable] = { 'stdT':0, 'stdF':0}
        for value, y in zip(self.rawDataDict[variable], self.rawDataDict[target]):
            if y == self.positivePred:
                v = (float(value) - self.averages[variable]['avgT'])
                self.stds[variable]['stdT'] +=  v * v
            else:
                v = (float(value) - self.averages[variable]['avgF'])
                self.stds[variable]['stdF'] += v * v

        # Divide by n - 1 for sample
        self.stds[variable]['stdT'] /= self.averages[variable]['countT'] - 1
        self.stds[variable]['stdF'] /= self.averages[variable]['countF'] - 1
        self.stds[variable]['stdT'] = math.sqrt(self.stds[variable]['stdT'])
        self.stds[variable]['stdF'] = math.sqrt(self.stds[variable]['stdF'])

    #####################        
    # rejectionSampling #
    #####################        
    def rejectionSampling(self, vars, query, q):
        '''
        Reproduce samples with the same distribution as
        the original distribution based upon CPTs
        '''
        start = time.time()
        # Move given/target to the back - order vars
        for index, var in enumerate(self.listVars):
            if var == self.given:
                self.listVars += [self.listVars.pop(index)]
                break
        samples = []
        
        targetCount = 0
        count = 0

        # we want a minimum of 100 samples
        while count < self.pSampleCount or targetCount < 100:#targetAchieved == False:
            sample = []

            # I don't think we need to sample the target=0 and =1 prior, but...I am :)
            target = '0'
            targetProb = self.learnt["P("+self.given+"='0')"][1]
            rnT = random.uniform(0,1)
            
            if rnT > 0.5: #targetProb:
                target = '1'  

            for variable in self.listVars:
                if variable == self.given: continue
                if variable not in vars: continue
                priorDict = {}
                for pProb in self.learnt.keys():
                    if "P(" + variable + "=" in pProb and "|"+self.given+"='" +target+"'" in pProb:
                        line = self.learnt[pProb]
                        priorDict[pProb] = line[1]
                # Sort key values - makes assigning random pick easier
                priorDict = {k: v for k, v in sorted(priorDict.items(), key=lambda item: item[1])}
                # Pick one of the atrributes as part of sample
                choose = []
                while len(choose) == 0:
                    rn = random.uniform(0,1)
                    lower = 0
                    for index,pProb in enumerate(priorDict.keys()):
                        if rn > lower and rn <= priorDict[pProb] + lower:# or index == len(priorDict.keys())-1:
                            var = pProb.split("|")[0].replace("P(","")
                            v = pProb.split("|")[0]
                            v = v.replace("P("+variable+"='",'').replace("'","")
                            choose.append(var)
                        lower = priorDict[pProb]
                
                # If 2 or more samples were picked - they have same prob
                # choose one of them
                if len(choose) != 0:
                    rn2 = random.randint(0, len(choose) - 1)
                    sample.append(choose[rn2])
                    count += 1

            sample.append(target)

            # Just check that we have enough samples that match - it is said
            # that 100 samples are needed
            found = True
            for v in sample:
                if v not in query:
                    found = False
                    break
            if found == True:
                targetAchieved = True
                targetCount += 1

            # A valid sample has been found
            if len(sample) == len(vars):     
                samples.append(sample)

        # MMMMmmmmmm..... I might be drunk :)
        T = 0; F = 0
        for s in samples:
            a = []
            for v in q.keys():
                a.append(v+"='"+q[v]+"'")
            t = a.copy(); f = a.copy(); tr = a.copy(); fr = a.copy(); tr.reverse()
            fr.reverse(); tr.append('1'); fr.append('0'); t.append('1'); f.append('0')
            if s == t or s == tr: T +=1
            if s == f or s == fr: F +=1
        
        end = time.time()
        self.show("Samples requested: " + str(self.pSampleCount) + " Samples needed: " +str(count))
        self.show("Counts     <"+self.given + "='" + self.positivePred + "'=" + str(T) + " , <"+self.given + "='0'=" + str(F) + ">")
        self.show("Normalised <"+self.given + "='" + self.positivePred + "'=" + str(T/(T+F)) + " , "+self.given + "='0'=" + str(F/(T+F)) + ">")
        self.show("Inference time: " + str(end - start) + " seconds")
        self.show()

    #################        
    # priorSampling #
    #################        
    def priorSampling(self):
        '''
        Reproduce samples with the same distribution as
        the original distribution based upon CPTs
        '''
        # Move given/target to the back - order vars
        for index, var in enumerate(self.listVars):
            if var == self.given:
                self.listVars += [self.listVars.pop(index)]
                break
        samples = []
        count = 0
        # We need a certain amount of samples
        while len(samples) != self.pSampleCount:
            sample = []
            for variable in self.listVars:
                if variable == self.given: continue
                priorDict = {}
                # Pick target
                rn = random.uniform(0,1)
                target = '0'
                if rn >= 0.5:
                    target = '1'

                for pProb in self.learnt.keys():
                    if "P(" + variable + "=" in pProb and "|"+self.given+"='"+target in pProb:
                        line = self.learnt[pProb]
                        priorDict[pProb] = line[1]
                # Sort key values - makes assigning random pick easier
                priorDict = {k: v for k, v in sorted(priorDict.items(), key=lambda item: item[1])}

                # Pick one of the atrributes as part of sample
                choose = []
                while len(choose) == 0:
                    rn = random.uniform(0,1)
                    for index,pProb in enumerate(priorDict.keys()):
                        if rn < priorDict[pProb]: # replace target=
                            v = pProb.replace("P("+variable+"='",'').replace("'|" + self.given + "='" + target + "')", '')
                            choose.append(v)
                # If 2 or more samples were picked - they have same prob
                # choose one of them
                if len(choose) != 0:
                    rn2 = random.randint(0, len(choose) - 1)
                    sample.append(choose[rn2])
            # A valid sample has been found
            if len(sample) == len(self.listVars) - 1:    
                sample.append(target)    
                samples.append(sample)
        # Write samples to CSV file - we can then test against this
        with open(self.outFile + "-priors.csv", 'w') as fp:
            for index, variable in enumerate(self.listVars):
                fp.write(variable)
                if index < len(self.listVars) - 1:
                    fp.write(',')
            fp.write('\n')
            for sample in samples:
                for index, attribute in enumerate(sample):
                    fp.write(attribute)
                    if index < len(sample) - 1:
                        fp.write(',')
                fp.write('\n')
            
    ###############
    # answerQuery #
    ###############
    def answerQuery(self,i, query):
        '''
        Main function that takes each line of a test file
        (which I call questions / queries) and produces an
        answer for each row. Discrete values have been
        previously learnt.
        '''
        self.show()
        # Log and standard have different math operations
        char = " * "    
        if self.commonSettings['log'] == 'True':
            char = " + "

        for question in self.questions:
            start = time.time()
            answers = self.getAnswers(question)
            self.displayAnswers(answers, char)
            results = self.enumerateAnswers(answers, char)
            results = self.constructResult(results)
            results = self.normaliseResults(results)
            prediction, probability = self.argMaxPrediction(results)
            end = time.time()
            self.show()
            self.show(self.given + "=" + str(prediction) + " with a prob of " + str(round(probability,self.dp)))
            self.show("Inference time: " + str(end - start))
            self.show()

    ############
    # readFile #
    ############
    def readFile(self, fileName):
        '''
        This was originally done using Pandas but replaced with my this version
        when it was thought that the pandas library use was not allowed.
        '''
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
        '''
        Create all structures keeping variables in order. Use all variables
        then remove one and test network. Put that variable back and remove another,
        repeat by removing 2 and then 3, 4 5 variables etc. May produce thousands
        of different structures.
        '''
        combos = []
        if self.findBestStrure == 'True':
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

    ########
    # show #
    ########
    def showS(self, line = '', endLine = '\n'):
        '''
        Either prints to screen, writes to a results file
        or both.
        '''

        if self.commonSettings['display'] == 'True':
            print(str(line), end=endLine)

        with open(self.outFile + "-results.txt", 'a') as f:
            f.write(str(line))
            f.write(endLine)

    ########
    # show #
    ########
    def showG(self, line = '', endLine = '\n'):
        '''
        Either prints to screen, writes to a results file
        or both.
        '''

        if self.commonSettings['display'] == 'True':
            print(str(line), end=endLine)

        with open(self.outFile + "-gaussian-results.txt", 'a') as f:
            f.write(str(line))
            f.write(endLine)

    #################
    # readQuestions #
    #################
    def readTestQueries(self):
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
                    qs = []
                    for index in indexes:
                        qs.append(data[index])
                    self.questions.append(qs)
                count += 1

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

    #################
    # getJointProbs #
    #################
    def getAnswers(self, question):
        '''
        Answers are in the form:
        P(target='0'|evidence) = P(target='0') + P(age='3'|target='0') + P(oldpeak='0'|target='0')
        P(target='1'|evidence) = P(target='1') + P(age='3'|target='1') + P(oldpeak='0'|target='1')
        they are in human readable format.
        This function prepares the answers.
        '''
        answers = {}
        for discrete in self.learnt.keys():
            if "P(" + self.given + "=" in discrete:
                answers[discrete] = [[discrete, self.learnt[discrete]]]
                try:
                    for index, option in enumerate(question):
                        if self.variables[index] != self.given:
                            answers[discrete].append(["P("+ self.variables[index] + "='" + option + "'|" + discrete[2:-1]+")", self.learnt["P("+ self.variables[index] + "='" + option + "'|" + discrete[2:-1]+")"]])
                except:
                    # This is a hack - but it works
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
        return prediction, value

    #####################
    # probOfTargetTruth #
    #####################
    def probOfTargetTruth(self, results, target):
        '''
        After normalising, what probability does the target have
        '''
        for index,key in enumerate(results.keys()):
            if key == "P(" + self.given + "='" + str(target) + "'|evidence)":
                return results[key][2]     

    #################
    # answerQueries #
    #################
    def answerQueries(self,i):
        '''
        Main function that takes each line of a test file
        (which I call questions / queries) and produces an
        answer for each row. Discrete values have been
        previously learnt.
        '''
        # For basic metrics
        
        self.bestResults[i] = {}
        self.bestResults[i]['Structure'] = self.listVars
        self.bestResults[i]['TotalQueries'] = 0
        self.bestResults[i]['Correct'] = 0
        self.bestResults[i]['Y_true'] = []
        self.bestResults[i]['Y_pred'] = []
        self.bestResults[i]['Y_prob'] = []
        self.bestResults[i]['Y_prob_pred'] = []

        meanInfTime = []

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

        # Start timer() to record inference time
        start = time.time()

        for question in self.questions:
            answers = self.getAnswers(question)
            self.displayAnswers(answers, char)
            results = self.enumerateAnswers(answers, char)
            results = self.constructResult(results)
            results = self.normaliseResults(results)
            prediction, probability = self.argMaxPrediction(results)

            # Make metrics        
            self.bestResults[i]['TotalQueries'] += 1
            self.bestResults[i]['Y_true'].append(question[qPosition])
            self.bestResults[i]['Y_pred'].append(prediction)
            self.bestResults[i]['Y_prob'].append(self.probOfTargetTruth(results, question[qPosition]))
            self.bestResults[i]['Y_prob_pred'].append(probability)

            # Within the lof file show predictions
            if prediction ==  question[qPosition]:
                self.bestResults[i]['Correct'] += 1
                self.show("Correct")
            else:
                self.show("Wrong")            
            self.show()

        # Record time taken
        end = time.time()
        meanInfTime.append(end-start)
        self.bestResults[i]['InferenceT'] = end - start

        self.bestResults[i]['balanced'] = balanced_accuracy_score(self.bestResults[i]['Y_true'], self.bestResults[i]['Y_pred']) 
        fpr, tpr, _ = roc_curve(self.bestResults[i]['Y_true'], self.bestResults[i]['Y_prob'], pos_label=self.positivePred)
        self.bestResults[i]['auc'] = auc(fpr, tpr)
        self.bestResults[i]['kl'] = self.KLDivergence(self.bestResults[i])
        self.bestResults[i]['brier'] = self.brier(self.bestResults[i])
        self.bestResults[i]['accuracy'] = self.bestResults[i]['Correct'] / len(self.bestResults[i]['Y_true'])
        self.bestResults[i]['LL'] = self.logLikelihood()
        self.bestResults[i]['BIC'] = self.baysianInfoCriterion(self.bestResults[i]['LL'])
        self.bestResults[i]['F1'] = f1_score(self.bestResults[i]['Y_true'], self.bestResults[i]['Y_pred'], pos_label=self.positivePred)
                
        # Confusion matrix
        try:
            tn, fp, fn, tp = confusion_matrix(self.bestResults[i]['Y_true'], self.bestResults[i]['Y_pred']).ravel()
            self.bestResults[i]['Confusion'] = {'TN': tn, "FP": fp, "FN": fn, "TP": tp}
            self.show("tn, fp, fn, tp: " + str(tn) + " " + str(fp) + " " + str(fn) + " " + str(tp))
        except:
            # We are here is there is not enough tests as in the 
            # play_tennis example
            self.bestResults[i]['Confusion'] = {}
            pass

        self.show(self.listVars)
        self.show("Correct predictions  : " + str(self.bestResults[i]['Correct']))
        self.show("Number of predictions: " + str(self.bestResults[i]['TotalQueries']))
        self.show("Balanced Accuracy    : " + str(round(self.bestResults[i]['balanced'] , self.dp)))
        self.show("Mean Inference time  : " + str(sum(meanInfTime)/len(meanInfTime)))
        self.show()

        # Is this the best structure?
        # Hill Climbing network structures
        if self.bestResults[i]['balanced'] > self.bestResults['BestAcc']:
            self.bestResults['BestAcc'] = self.bestResults[i]['balanced']  
            self.bestResults['BestStructure'] = self.listVars
            self.bestResults['BestStructureI'] = i
            self.showS("Structure #" + str(i) + " " * (17 - len("Structure #" + str(i) + " ")) + " : P(" + self.given + "|", "")
            for index,variable in enumerate(self.bestResults[i]['Structure']):
                #print(index,variable)
                if variable != self.given:
                    self.showS(variable, "")
                if index < len(self.bestResults[i]['Structure']) - 1 and index > 0:
                    self.showS(",", "")
            self.showS(")")
            self.showS("Log Likelihood   : " + str(round(self.bestResults[i]['LL'], self.dp)))
            self.showS("BIC score        : " + str(round(self.bestResults[i]['BIC'], self.dp)))
            self.showS("Test results")
            self.showS("------------")
            self.showS("Balanced Acc     : " + str(round(self.bestResults[i]['balanced'] * 100.0, self.dp)) + "%")
            self.showS("Std Accuracy     : " + str(round(self.bestResults[i]['accuracy'] * 100, self.dp)) + "%")
            self.showS("Area under curve : " + str(round(self.bestResults[i]['auc'], self.dp)))
            self.showS("KL divergence    : " + str(round(self.bestResults[i]['kl'], self.dp)))
            self.showS("Brier score      : " + str(round(self.bestResults[i]['brier'], self.dp)))
            self.showS("F1-score         : " + str(round(self.bestResults[i]['F1'], self.dp)))
            self.showS("Confusion matrix : " + str(self.bestResults[i]['Confusion']))
            self.showS("Inference time   : " + str(round(self.bestResults[i]['InferenceT'], self.dp)) + " seconds")
            self.showS()

    #################
    # logLikelihood #
    #################
    def logLikelihood(self):
        # Self.rawDataDict contains all training data
        # self.learnt contains all probabilities
        # self.listVars conatins the variables in THIS structure
        LL = 0
        for variable in self.listVars:
            for trainVariable in self.rawDataDict.keys():
                # if the variable is in this structure
                if variable in self.rawDataDict.keys() and variable == trainVariable:
                    for index,attribute in enumerate(self.rawDataDict[trainVariable]):
                        if variable != self.given:
                            # get the log(P(var=attrib|target=rawY)) from learnt
                            search = "P(" + variable + "='" + attribute + "'|" + self.given +"='"+str(self.rawDataDict[self.given][index]) + "')"
                            value = self.learnt[search][1]
                            LL += math.log(value)
                        elif variable == self.given:
                            # get log(P(given=rawY)) from learnt
                            search = "P(" + self.given +"='"+str(self.rawDataDict[self.given][index]) + "')"
                            value = self.learnt[search][1]
                            LL += math.log(value)
        return LL
        
    ########################
    # baysianInfoCriterion #
    ########################    
    # Taken from workshop material but modified to match own method.
    # Overall function simplified
    def baysianInfoCriterion(self, LL):
        penalty = 0
        for variable in self.listVars:
            # number of params = number of P(variable=attrib|given) or P(variable=attrib) if it is a predictor var  
            penalty += (math.log(self.total) * self.numberOfParams(variable)) / 2
        BIC = LL - penalty
        return BIC

    ##################
    # numberOfParams #
    ##################
    def numberOfParams(self, variable):
        '''
        number of params = number of P(variable=attrib|given) or P(variable=attrib) if it is a predictor var 
        '''
        count = 0

        # It has to be one or the other - we know variable and number
        # of attributes is in the learnt dict. If we can't match on first
        # count it must be a predictor so count again without the pipe |

        # Presume a non target variable
        for CP in self.learnt:
            if "P(" + variable + "='" in CP and "|" in CP:
                count += 1
        
        # Must be the target
        if count == 0:
            for CP in self.learnt:
                if "P(" + variable + "='" in CP and "|" not in CP:
                    count += 1            
        
        return count

    ################
    # KLDivergence #
    ################
    # Taken from workshop example code but modified
    def KLDivergence(self,results):
        '''
        Calculate KL divergence
        '''
        Y_true = self.convertBinary(results['Y_true'])
        # KLConstant avoids NaN    
        P = np.asarray(Y_true) + self.KLConstant
        Q = np.asarray(results['Y_prob']) + self.KLConstant
        return np.sum(P * np.log(P/Q))

    #########
    # brier #
    #########
    def brier(self, results):
        Y_true = self.convertBinary(results['Y_true'])
        return brier_score_loss(Y_true, results['Y_prob'])
     
    #################
    # convertBinary #
    #################
    def convertBinary(self, array):
        '''
        Takes the positive attribute and converts to binary 1.
        All other values are 0
        '''
        Y_true = []

        # Converts yes / no in to 1 / 0
        for yt in array:
            if yt == self.positivePred:
                Y_true.append(1)
            else:
                Y_true.append(0)

        return Y_true

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
            fp.write("Conditional Probability Tables\n")
            fp.write("------------------------------\n")
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
        self.show()
        self.show("Conditional Probability Tables")
        self.show("------------------------------")
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
                            elif self.avoidZeros == 'True':
                                self.learnt["P(" + str(variable) + "='" + str(attribute) + "'|" + str(self.given) + "='" + str(givenOption) + "')" ] = (str(countXY) + "/" + str(countY), float((countXY + float(self.commonSettings['simple'])) / countY))
                            elif self.avoidZeros == 'False':
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
        '''
        How many occurances of attribute within the variable?
        '''
        count = 0
        for attrib in self.rawDataDict[variable]:
            if attrib == match:
                count += 1
        return count

    #############################
    # countMatchingAttribsMulti #
    #############################
    def countMatchingAttribsMulti(self, variable1, variable2, match1, match2):
        '''
        How many occurances where atrib1 and attrib2 match the given values?
        '''
        count = 0
        for (attrib1, attrib2) in zip(self.rawDataDict[variable1], self.rawDataDict[variable2]):
            if attrib1 == match1 and attrib2 == match2:
                count += 1
        return count

    #######################
    # getVariableContents #
    #######################
    def getVariableContents(self,listOfVars):
        '''
        Return a list of variables. This is a hack that was needed
        when pandas dataframes were replaced by my own code.
        '''
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
        Actually displays / logs Priors.
        '''
        # NOTE change the word discrete with Prior
        self.show("Discrete variables and probabilities for: " + str(self.fileName))
        self.show("Total samples:" + str(self.total))
        self.show()
        for variable in self.discretes:
            self.show("Variable:" + str(variable))
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
                    else:
                        self.learnt["P(" + str(variable) + "='" + str(attribute) + "')"] = ("1/"+str(self.total)), float(1/self.total)
                else:
                    self.discretes[variable][attribute]['total'] += 1
                    countX = self.discretes[variable][attribute]['total']
                    domainSizeX = len(self.discretes[variable])
                    self.discretes[variable][attribute]['prob'] = (countX + 1) / (self.total + domainSizeX)
                    if self.MLE == 'True':
                        self.learnt["P("+str(variable)+"='"+str(attribute)+"')"] = (str(countX + 1) + "/"+ str(self.total + domainSizeX)),float((countX + 1) / (self.total + domainSizeX))
                    else:
                        self.learnt["P("+str(variable)+"='"+str(attribute)+"')"] = (str(countX) + "/"+ str(self.total)),float((countX) / self.total)


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
    
    # the exception will catch index errors
    for n in range(1, 20):
        try:
            # Queries for this network
            q = [queries[key] for key in queries.keys() if "query"+str(n) in key]
        
            # Learn and save results
            NB = NaiveBayes(bayesConfig, n, False, common)
            # Test and save results
            NB = NaiveBayes(bayesConfig, n, True, common)
            # Run queries
            NB = NaiveBayes(bayesConfig, n, True, common, q)
        # Only need except due to for loop
        except KeyError as e:
            print("All tests have been run. Please see results folder.")
            quit()


##################
# start properly #
##################
if __name__ == "__main__":
   main(sys.argv[1:])