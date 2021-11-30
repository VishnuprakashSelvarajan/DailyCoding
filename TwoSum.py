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


def bubbleSort(arr):
    n = len(arr)

    # Traverse through all array elements
    for i in range(n - 1):
        # range(n) also work but outer loop will repeat one time more than needed.

        # Last i elements are already in place
        for j in range(0, n - i - 1):

            # traverse the array from 0 to n-i-1
            # Swap if the element found is greater
            # than the next element
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]


arr = [64, 34, 25, 12, 22, 11, 90]

print(bubbleSort(arr))

# Python program for implementation of Quicksort Sort

# This function takes last element as pivot, places
# the pivot element at its correct position in sorted
# array, and places all smaller (smaller than pivot)
# to left of pivot and all greater elements to right
# of pivot


def partition(arr, low, high):
    i = (low-1)		 # index of smaller element
    pivot = arr[high]	 # pivot

    for j in range(low, high):

        # If current element is smaller than or
        # equal to pivot
        if arr[j] <= pivot:

            # increment index of smaller element
            i = i+1
            arr[i], arr[j] = arr[j], arr[i]

    arr[i+1], arr[high] = arr[high], arr[i+1]
    return (i+1)

# The main function that implements QuickSort
# arr[] --> Array to be sorted,
# low --> Starting index,
# high --> Ending index

# Function to do Quick sort


def quickSort(arr, low, high):
    if len(arr) == 1:
        return arr
    if low < high:

        # pi is partitioning index, arr[p] is now
        # at right place
        pi = partition(arr, low, high)

        # Separately sort elements before
        # partition and after partition
        quickSort(arr, low, pi-1)
        quickSort(arr, pi+1, high)


# Driver code to test above
arr = [10, 7, 8, 9, 1, 5]
n = len(arr)
quickSort(arr, 0, n-1)
print("Sorted array is:")
for i in range(n):
    print("%d" % arr[i])
