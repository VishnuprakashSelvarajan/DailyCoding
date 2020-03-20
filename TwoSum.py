'''
Two Sum
'''

def twoSum(numbers, target):
    seen = set()
    output = set()
    index = 0
    for index, num in enumerate(numbers):
        diff = target - num
        if diff not in seen:
            seen.add(num)
        else:
            output.add((min(index+1, numbers.index(diff)+1), max(index+1, numbers.index(diff)+1)))

    return list(output)

print(twoSum([2,2,7,7,11,15], 9))
