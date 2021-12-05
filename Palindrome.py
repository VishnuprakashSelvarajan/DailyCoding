def IsPalindrome(text):

    if len(text) == 2:
        return [True if text[0] == text[1] else False]
    if len(text) < 2:
        return True



    text = list(text)
    while len(text) > 1:
        if text[0] == text[-1]:
            text.pop(0)
            text.pop(-1)
            IsPalindrome(text)
        else:
            return False


    return True

print(IsPalindrome('aaaaaabb'))

def powerset(arr):

    def all_subset(arr, subset):
        while len(arr) > 0:
            subset.append(arr[0:len(arr)])
            arr.pop(-1)
            powerset(arr)
        return subset

    subsets = []
    for i in range(0,len(arr)):
        subsets.append(all_subset(arr[i:len(arr)], []))

    return subsets
print(powerset([1,2,3,4,5,6,7,8,9]))

