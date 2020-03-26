"""
Given a list of integers S and a target number k, write a function that returns a subset of S that adds up to k.
If such a subset cannot be made, then return null.

Integers can appear more than once in the list. You may assume all numbers in the list are positive.

For example, given S = [12, 1, 61, 5, 9, 2] and k = 24, return [12, 9, 2, 1] since it sums up to 24.
"""
def SumOfSubset(S, k):
    # First find all the subsets of the given array S
    subset = [0 for i in range(len(S))]
    #Define a helper function to find all the subsets
    helper(S, subset, 0, k)

def helper(S, subset, i, k):
    if i == len(S):
        if sum(subset) == k:
            print(subset)
    else:
        subset[i] = 0
        helper(S, subset, i + 1, k)
        subset[i] = S[i]
        helper(S, subset, i + 1, k)

SumOfSubset([12, 1, 61, 5, 9, 2], 24)

""" Another Solution"""
def SubsetSumtoTarget(S, k):
    return recur(S, k, len(S)-1)

# Returns the number of subsets present in S which is equal to the target k

def recur(S, k, index):
    if k == 0:
        return 1
    elif k < 0:
        return 0
    elif index < 0:
        return 0
    elif k < S[index]: # if the target is less than item in index
        return recur(S, k , index-1)
    else:
        return recur(S, k-S[index], index-1) + recur(S, k, index-1)

print(SubsetSumtoTarget([12, 1, 61, 5, 9, 2], 24))


