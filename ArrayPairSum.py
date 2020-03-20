'''
Array Pair Sum
Given an integer array, output all the unique pairs that sum up to a specific value k
So the input:
pair_sum([1,3,2,2], 4)
Output:
(1,3)
(2,2)
'''
def pair_sum(array, k):
    if len(array) < 2:
        return('Too Small')

    seen = set()
    output= set()
    for num in array:
        target = k - num
        if target not in seen:
            seen.add(num)
        else:
            output.add((min(num, target), max(num, target)))

    return output

print(pair_sum([1,3,2,0,4,5,-1,-1,2,1,5,3,1,1,2,3], 4))
#####################