# NOTE Taken from https://www.geeksforgeeks.org/print-all-possible-combinations-of-r-elements-in-a-given-array-of-size-n/
# with some modification by myself, Andy Perrett
# It has been objectified and made to return all combos

# Program to print all combination
# of size r in an array of size n
 
# The main function that prints all
# combinations of size r in arr[] of
# size n. This function mainly uses
# combinationUtil()

#########
# class #
#########
class Combinations:

    ############
    # __init__ #
    ############
    def __init__(self, structure, s):
        # c will be returned with a list of combinations
        self.c = []
        n = len(structure)
        self.getCombination(structure, n, s)
        
    ###################
    # getCombinations #
    ###################
    def getCombinations(self):
        return self.c

    ##################
    # getCombination #
    ##################
    def getCombination(self, arr, n, r):
    
        # A temporary array to store
        # all combination one by one
        data = [0] * r
        
        # Print all combination using
        # temporary array 'data[]'
        self.combinationUtil(arr, n, r, 0, data, 0)
        return 
    
    ###################
    # combinationUtil #
    ###################
    def combinationUtil(self,arr, n, r, index, data, i):
        ''' arr[] ---> Input Array
        n     ---> Size of input array
        r     ---> Size of a combination to be printed
        index ---> Current index in data[]
        data[] ---> Temporary array to store
                    current combination
        i     ---> index of current element in arr[]     '''
        # Current combination is ready,
        # print it
        # Andy modified this part to append this result to the previous
        if (index == r):
            result = []
            for j in range(r):
                result.append(data[j])
            self.c.append(result)

            return
    
        # When no more elements are
        # there to put in data[]
        if (i >= n):
            return 
    
        # current is included, put
        # next at next location
        data[index] = arr[i]
        self.combinationUtil(arr, n, r, index + 1, data, i + 1)
    
        # current is excluded, replace it
        # with next (Note that i+1 is passed,
        # but index is not changed)
        self.combinationUtil(arr, n, r, index, data, i + 1)
    

# This code is contributed
# by ChitraNayal