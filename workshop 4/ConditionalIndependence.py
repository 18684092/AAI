#############################################################################
# ConditionalIndependence.py
#
# Implements functionality for conditional independence tests via the
# library causal-learn (https://github.com/cmu-phil/causal-learn), which
# can be used to identify edges to keep or remove in a graph given a dataset.
#
# This requires installing the following (at Uni-Lincoln computer labs):
# 1. Type Anaconda Prompt in your Start icon
# 2. Open your terminal as administrator
# 3. Execute=> pip install causal-learn
#
# At the bottom of this file are the USAGE instructions to run this program.
#
# Version: 1.0, Date: 19 October 2022
# Contact: hcuayahuitl@lincoln.ac.uk
#############################################################################

import sys
from causallearn.utils.cit import CIT


class ConditionalIndependence:
    chisq_obj = None
    rand_vars = []
    rv_all_values = []

    def __init__(self, file_name):
        data = self.read_data(file_name)
        self.chisq_obj = CIT(data, "chisq")

    def read_data(self, data_file):
        print("\nREADING data file %s..." % (data_file))
        print("---------------------------------------")

        with open(data_file) as csv_file:
            for line in csv_file:
                line = line.strip()
                if len(self.rand_vars) == 0:
                    self.rand_vars = line.split(',')
                else:
                    values = line.split(',')
                    self.rv_all_values.append(values)

        print("RANDOM VARIABLES=%s" % (self.rand_vars))
        print("VARIABLE VALUES (first 10)=%s" % (self.rv_all_values[:10])+"\n")
        return self.rv_all_values

    def parse_test_args(self, test_args):
        main_args = test_args[2:len(test_args)-1]
        variables = main_args.split('|')[0]
        Vi = variables.split(',')[0]
        Vj = variables.split(',')[1]
        parents_i = []
        for parent in (main_args.split('|')[1].split(',')):
            if parent.lower() == 'none':
                continue
            else:
                parents_i.append(parent)

        return Vi, Vj, parents_i

    def get_var_index(self, target_variable):
        for i in range(0, len(self.rand_vars)):
            if self.rand_vars[i] == target_variable:
                return i
        print("ERROR: Couldn't find index of variable "+str(target_variable))
        return None

    def get_var_indexes(self, parent_variables):
        if len(parent_variables) == 0:
            return None
        else:
            index_vector = []
            for parent in parent_variables:
                index_vector.append(self.get_var_index(parent))
            return index_vector

    def compute_pvalue(self, variable_i, variable_j, parents_i):
        var_i = self.get_var_index(variable_i)
        var_j = self.get_var_index(variable_j)
        par_i = self.get_var_indexes(parents_i)
        p = self.chisq_obj(var_i, var_j, par_i)

        print("X2test: Vi=%s, Vj=%s, pa_i=%s, p=%s" %
              (variable_i, variable_j, parents_i, p))
        return p


if len(sys.argv) != 3:
    print("USAGE: ConditionalIndepencence.py [train_file.csv] [I(Vi,Vj|parents)]")
    print("EXAMPLE1: ConditionalIndepencence.py play_tennis-train-x3.csv 'I(O,PT|None)'")
    print("EXAMPLE2: ConditionalIndepencence.py play_tennis-train-x3.csv 'I(PT,T|O)'")
    print("EXAMPLE3: ConditionalIndepencence.py play_tennis-train-x3.csv 'I(PT,T|O,H)'")
    exit(0)
else:
    data_file = sys.argv[1]
    test_args = sys.argv[2]

    ci = ConditionalIndependence(data_file)
    Vi, Vj, parents_i = ci.parse_test_args(test_args)
    ci.compute_pvalue(Vi, Vj, parents_i)
