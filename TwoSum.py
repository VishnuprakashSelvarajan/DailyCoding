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
            output.add((min(index + 1, numbers.index(diff) + 1), max(index + 1, numbers.index(diff) + 1)))
    return list(output)


# print(twoSum([2,2,7,7,11,15], 9))

def merge_interval(inervals):
    if len(intervals) == 1:
        return intervals

    intervals.sort()
    result = [intervals[0]]
    for i in range(1, len(intervals)):
        if result[-1][1] >= intervals[i][0]:
            result[-1] = [min(result[-1][0], intervals[i][0]), max(result[-1][1], intervals[i][1])]
        else:
            result.append(intervals[i])
    return result

def insert_intervals(intervals, newInterval):

    intervals.append(newInterval)
    if len(intervals) == 1:
        return intervals

    intervals.sort()
    result = [intervals[0]]
    for i in range(1, len(intervals)):
        if result[-1][1] >= intervals[i][0]:
            result[-1] = [min(result[-1][0], intervals[i][0]), max(result[-1][1], intervals[i][1])]
        else:
            result.append(intervals[i])
    return result

print(insert_intervals([[3,5],[12,15]], [6,6]))

def merge(intervals, newInterval):
    result = []
    if intervals == []:
        result.append(newInterval)
        return result
    if intervals[-1][1] < newInterval[0]:
        intervals.append(newInterval)
        return intervals
    if intervals[0][0] > newInterval[1]:
        result.append(newInterval)
        for i in intervals:
            result.append(i)
        return result

    result = [intervals[0]]
    for i in range(0, len(intervals)+1):
        if result[-1][1] < newInterval[0] and newInterval[1] < intervals[i][0]:
            result.append(newInterval)
            break
        elif result[-1][1] >= newInterval[0]:
            result[-1] = [min(result[-1][0], newInterval[0]), max(result[-1][1], newInterval[1])]
            break
        else:
            if intervals[i] not in result:
                result.append(intervals[i])

    if i != len(intervals):
        for j in range(i, len(intervals)):
            if result[-1][1] >= intervals[j][0]:
                result[-1] = [min(result[-1][0], intervals[j][0]), max(result[-1][1], intervals[j][1])]
            else:
                result.append(intervals[j])

    return result


print(merge([[3,5],[12,15]], [6,6]))

def insertion_sort(arr):

    #Traverse through the array
    for i in range(1, len(arr)):

        key = arr[i]

        # Move elements of arr[0..i-1], that are
        # greater than key, to one position ahead
        # of their current position
        j = i-1
        while j >= 0 and key < arr[j]:
            arr[j+1]  = arr[j]
            j -= 1
        arr[j+1] = key



print(insertion_sort([12, 11, 13, 5, 6]))