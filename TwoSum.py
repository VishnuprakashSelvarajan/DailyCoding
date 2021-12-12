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

# Case 1: M ~ N
def findDuplicates(arr1, arr2):
    duplicates = set()
    i,j = 0,0
    while i < len(arr1) and j < len(arr2):
        if arr1[i] == arr2[j]:
            duplicates.add(arr1[i])
            i+=1
            j+=1
        elif arr1[i] < arr2[j]:
            i+=1
        else:
            j+=1

    return duplicates

print(findDuplicates([1,2,3,5,6,7,20], [3,6,7,8,20,7]))

# Case 2: M >> N
def findDuplicatesBinary(arr1, arr2):
    duplicates = []
    for number in arr1:
        if binarySearch(number, arr2) != -1:
            duplicates.append(number)

    return duplicates

def binarySearch(number, arr2):
    start = 0
    end = len(arr2)-1
    while start <=end:
        mid = start + int((end-start)/2)
        if number > arr2[mid]:
            start = mid + 1
        elif number == arr2[mid]:
            return number
        else:
            end = mid - 1
    return -1

print(findDuplicatesBinary([1,2,3,5,6,7,20], [3,6,7,8,20,7]))


def find_first(array, num):

    start = 0
    end = len(array)-1
    index = -1
    while start < end:
        mid = start + int((end-start)/2)
        if num > array[mid]:
            start = mid +1
            continue
        elif num == array[mid]:
            index = mid
            end = mid
            continue
        else:
            end = mid
            continue

    return index

print(find_first([200,200,200,200,200,200,200,200,200,500,500,500,500,600], 600))

def mergeIntervals(intervals):

    intervals.sort()
    if intervals == []:
        return []
    result = [intervals[0]]
    for i in range(0,len(intervals)):
        if result[-1][1] >= intervals[i][0]:
            result[-1] = [min(result[-1][0], intervals[i][0]), max(result[-1][1], intervals[i][1])]
        else:
            result.append(intervals[i])

    return result

print(mergeIntervals([[2,5],[2,10],[23,23],[34,56]]))


def inorderTraversal_2(root):
    stack, result = [], []
    node = root
    while node or stack:
        while node:
            stack.append(node)
            node = node.left

        node = stack.pop()
        result.append(node.val)

        node = node.right

    return result

def inorderTraversal(root):

    result = []

    def inorder(root):
        if root:
            inorder(root.left)
            result.append(root.val)
            inorder(root.right)

    inorder(root)
    return result


def preorderTraversal(root):

    result = []
    def preorder(root):
        if root:
            result.append(root.val)
            preorder(root.left)
            preorder(root.right)

    preorder(root)
    return result

def postorderTraversal(root):

    value = []

    def postorder(root):
        if root:
            postorder(root.left)
            postorder(root.right)
            value.append(root.val)

    postorder(root)
    return value

def levelOrder(root):
    if not root:
        return []
    out = []
    parents = [root]
    children = []
    level = []
    while len(parents) > 0:
        parent = parents.pop(0)
        if parent.left:
            children.append(parent.left)
        if parent.right:
            children.append(parent.right)
        level.append(parent.val)
        if len(parents) == 0:
            parents = children
            children = []
            out.append(level)
            level = []
    return out

def test(nums):

    def squares(a):
        return a*a
    def even(a):
        return a%2 ==0

    #return list((map(squares, nums)))
    #return list((map(lambda x: x**2, nums)))
    return list(filter(lambda x: x%2==0, nums))
    #return list(filter(even, nums))

print(test([2,3,4,5,6,7,8,9]))

memo = {}
def fibnocci(n):
    if n in memo:
        return memo[n]

    if n == 0 or n == 1:
        return n

    output = fibnocci(n-2) + fibnocci(n-1)
    memo[n] = output
    return output

print(fibnocci(10))

def fibnocci_nth(n):

    if n == 0 or n ==1:
        return 1

    output = [1, 1]
    for i in range(2,n):
        output.append(output[i-2] + output[i-1])

    return output

print(fibnocci_nth(20))


def hasPathSum(root, targetSum):
    res = []

    def dfs(node, path):

        if node is None:
            return

        path.append(int(node.val))
        dfs(node.left, path)
        dfs(node.right, path)

        if node.left is None and node.right is None:
            res.append(sum(path))

        del path[-1]

    dfs(root, [])
    if targetSum in res:
        return True

    return False


def flatten_dict(init, lkey=''):
    ret = {}
    for rkey,val in init.items():
        key = lkey+rkey
        if isinstance(val, dict):
            ret.update(flatten_dict(val, key+'.'))
        else:
            ret[key] = val
    return ret

nums = {
            "Key1" : "1",
            "Key2" : {
                "a" : "2",
                "b" : "3",
                "c" : {
                    "d" : "3",
                    "e" : {
                        "" : "1"
                    }
                }
            }
        }
print(flatten_dict(nums))

def basic_regexp(text, pattern):
    text = list(text)
    pattern = list(pattern)
    i = 0
    while len(text) > 0:
        if len(pattern) <= 0:
            return False
        print(text)
        if text[0] == pattern[0] or pattern[0] == '*':
            text.pop(0)
            pattern.pop(0)
        elif pattern[0] == '.':
            text.pop(0)
        else:
            return False
    if text:
        return False

    return True

print(basic_regexp("abb", "a.*"))


def maximum_path_sum(node):
    res = []

    def depth(node, path):
        if node == None:
            return
        path.append(node.val)
        depth(node.left)
        depth(node.right)
        if node.left == None and node.right == None:
            res.append(sum(path))
        del path[-1]

    depth(node, [])
    return max(res)
