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

print(twoSum([2,2,7,7,11,15], 9))

def merge_interval(intervals):
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

def is_isogram(string):
    string = string.lower()
    seen = []
    for i in string:
        if i.isalpha() and i not in seen:
            seen.append(i)
        elif not i.isalpha():
            continue
        else:
            return False

    return True

print(is_isogram('Alphabet'))


def minWindow(s,t):

    if len(t) > len(s):
        return ""

    min_substring = s
    max = len(s)

    def find_item(s_sub, t):
        for i in t:
            if t.count(i) > s_sub.count(i):
                return 0
        return 1

    for i in range(0, len(s)):
        substr = s[i:max]
        if find_item(substr, t):
            if len(substr) < len(min_substring):
                min_substring = substr
        else:
            break
        # Find if we reduce right side get to a more minimum sub string
    print(min_substring)
    for i in range(len(min_substring), 0, -1):
        substr = min_substring[0:i]
        if find_item(substr, t):
            if len(substr) < len(min_substring):
                min_substring = substr
        else:
            break

    return min_substring

print(minWindow('aabc', 'b'))


def count_words(sentence):
    sentence = sentence.replace(",", " ")
    sentence = sentence.replace("\n", " ")
    separated = sentence.split(" ")
    import re
    seen = {}
    for word in separated:
        if re.search(r'\'?\"?(\S+)', word):
            word = word.lower()
            out = re.findall(r'([0-9a-zA-Z]+)', word)
            if len(out) == 2 and "'" in word:
                word = out[0]+"'"+out[1]
                if word not in seen:
                    seen[word] = 1
                else:
                    seen[word] += 1
            else:
                for word in out:
                    if word not in seen:
                        seen[word] = 1
                    else:
                        seen[word] += 1

    return seen
print(count_words("hey,my_spacebar_is_broken, this's"))

import re
def abbreviate(words):
    words = words.split(" ")
    acronyum = ''
    for w in words:
        if re.findall(r'[a-zA-Z]', w):
            word = re.findall(r'([a-zA-Z]+)', w)
            print(w)
            if '-' in w:
                acronyum+=word[0][0]
                acronyum+=word[1][0]
            else:
                acronyum+=word[0][0]
    acronyum = acronyum.upper()
    return acronyum

print(abbreviate("Complementary metal-oxide semiconductor"))


def valid(card_num):

    card_num = card_num.replace(" ", "")
    if len(card_num) <= 1:
        return False
    if not card_num.isdecimal():
        return False
    final_card_num = [0 for i in range(len(card_num))]
    even_nums = [i for i in range(len(card_num), -1, -2)]
    del even_nums[0]
    for index in even_nums:
        final_card_num[index] = int(card_num[index]) * 2 if int(card_num[index]) * 2 < 9 else int(card_num[index]) * 2 - 9
    odd_nums = [i for i in range(0, len(card_num), 2)]
    for index in odd_nums:
        final_card_num[index] = int(card_num[index])

    total_card_sum = sum(final_card_num)
    print(total_card_sum)

    return True if total_card_sum % 10 == 0 else False

print(valid("095 245 88"))

def find_hour(hr):
    if hr < 24:
        return hr
    else:
        return find_hour(hr - 24)

print(find_hour(121))


def calculate_time(hr, min):
    hour = int(hr)
    minute = int(min)
    # add hour if minute goes beyond 60
    hour = hour + (minute // 60)
    minute = (minute % 60)

    def find_hour(hr):
        if hr < 24:
            return hr
        else:
            return find_hour(hr - 24)

    def find_hour_neg(hr):
        if hr < 24 and hr >= 0:
            return hr
        else:
            return find_hour(24 - hr)

    # If hour is more than 24 revert it back
    # Roll over if the hour  is in negative
    if hour >= 0:
        hour = find_hour(hour)
    else:
        hour = find_hour_neg(hour)
        hour = 24 - hour
    return [hour, minute]


class Clock:
    def __init__(self, hour, minute):
        self.hour = int(hour)
        self.minute = int(minute)

        output = calculate_time(self.hour, self.minute)
        self.hour = output[0]
        self.minute = output[1]

    def __repr__(self):
        if self.hour < 10:
            hour = "0" + str(self.hour)
        else:
            hour = str(self.hour)

        if self.minute < 10:
            minute = "0" + str(self.minute)
        else:
            minute = str(self.minute)

        time = hour + ":" + minute
        return time

    def __eq__(self, other):
        pass

    def __add__(self, minutes):
        self.minute = self.minute + int(minutes)
        output = calculate_time(self.hour, self.minute)
        self.hour = output[0]
        self.minute = output[1]
        if self.hour < 10:
            hour = "0" + str(self.hour)
        else:
            hour = str(self.hour)

        if self.minute < 10:
            minute = "0" + str(self.minute)
        else:
            minute = str(self.minute)

        time = hour + ":" + minute
        return time

    def __sub__(self, minutes):

        self.minute = self.minute - int(minutes)

        output = calculate_time(self.hour, self.minute)
        self.hour = output[0]
        self.minute = output[1]
        if self.hour < 10:
            hour = "0" + str(self.hour)
        else:
            hour = str(self.hour)

        if self.minute < 10:
            minute = "0" + str(self.minute)
        else:
            minute = str(self.minute)

        time = hour + ":" + minute
        return time

print(Clock(2, 20) - 3000)


def tally(rows):
    # Initial title create
    output = ["Team                           | MP |  W |  D |  L |  P"]
    results_db = {}

    if rows == []:
        return output

    for row in rows:
        row = row.split(';')
        team = row[0]
        team2 = row[1]
        if team in results_db:
            results_db[team]['mp'] += 1
        else:
            results_db[team] = {}
            results_db[team]['mp'] = 1
            results_db[team]['w'] = 0
            results_db[team]['d'] = 0
            results_db[team]['l'] = 0

        if team2 in results_db:
            results_db[team2]['mp'] += 1
        else:
            results_db[team2] = {}
            results_db[team2]['mp'] = 1
            results_db[team2]['w'] = 0
            results_db[team2]['d'] = 0
            results_db[team2]['l'] = 0

        if row[2] == 'win':
            results_db[team]['w'] += 1
            results_db[team2]['l'] += 1
        elif row[2] == 'draw':
            results_db[team]['d'] += 1
            results_db[team2]['d'] += 1
        else:
            results_db[team]['l'] += 1
            results_db[team2]['w'] += 1
        results_db[team]['p'] = 3 * results_db[team]['w'] + results_db[team]['d']
        results_db[team2]['p'] = 3 * results_db[team2]['w'] + results_db[team2]['d']

        # Populate the table
    output = "Team                           | MP |  W |  D |  L |  P\n"

    top = '{t:<31}|{mp:^4}|{w:>3} |{d:>3} |{l:>3} |{p:>3}{n}'
    res = top.format(t='Team', mp='MP', w='W', d='D', l='L', p='P', n='\n')

    for team in results_db:
        top = '{t:<31}|{mp:^4}|{w:>3} |{d:>3} |{l:>3} |{p:>3}{n}'
        res += top.format(t=team, mp=results_db[team]['mp'], w=results_db[team]['w'], d=results_db[team]['d'],
                          l=results_db[team]['l'], p=results_db[team]['p'], n='\n')

        # entry = "{} |  {} |  {} |  {} |  {} |  {}\n".\
        # format(team.ljust(30, ' '), results_db[team]['mp'], results_db[team]['w'], results_db[team]['d'],\
        # results_db[team]['l'], results_db[team]['p'])
        # output += entry

    # output = output.splitlines()
    return res

print(tally([
    "Allegoric Alaskans;Blithering Badgers;win",
    "Devastating Donkeys;Courageous Californians;draw",
    "Devastating Donkeys;Allegoric Alaskans;win",
    "Courageous Californians;Blithering Badgers;loss",
    "Blithering Badgers;Devastating Donkeys;loss",
    "Allegoric Alaskans;Courageous Californians;win",
]))

def num_islands(grid):
    result = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 1:
                result += 1
                helper(i, j, grid)
    return result

def helper(row, column, grid):
    if grid[row][column] != 1:
        return
    grid[row][column] = 0

    #Case1: Moving Up
    if row > 0:
        helper(row-1, column, grid)

    #Case2: Moving down
    if row < len(grid)-1:
        helper(row+1, column, grid)

    #Case3: Moving left
    if column > 0:
        helper(row, column-1, grid)

    #Case4: Moving right
    if column < len(grid[0])-1:
        helper(row, column+1, grid)

example = [[0,1,0,1,0],
[0,0,1,1,1],
[1,0,0,1,0],
[0,1,1,0,0],
[1,0,1,0,1]]
print(num_islands(example))


def is_pangram(sentence):
    letters_to_be_seen = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
                          'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

    sentence = sentence.replace(" ", "")
    for letter in sentence:
        letter = letter.lower()
        if letter in letters_to_be_seen:
            letters_to_be_seen.remove(letter)

    if len(letters_to_be_seen) == 0:
        return True
    else:
        return False

print(is_pangram('"Five quacking Zephyrs jolt my wax bed."'))


def findMissingRanges(nums, lower, upper):
    output = []
    if len(nums) == 0:
        if lower == upper:
            return [str(lower)]
        else:
            output.append(str(lower) + "->" + str(upper))
            return output

    missing = []
    def find_range(start, end, lower, upper):
        if start <= upper:
            if end != start+1 and end <= upper and start >= lower:
                return [start+1, end-1]
            elif end != start+1 and start >= lower and end > upper:
                return [start+1, upper]
            else:
                return 0
        else:
            return 0

    start = 0
    end = start + 1
    while end <= len(nums)+1:
        if end < upper:
            if find_range(nums[start], nums[end], lower, upper):
                missing.append(find_range(nums[start], nums[end], lower, upper))
            start+=1
            end += 1
        else:
            break
    if nums[-1] < upper:
        missing.append([nums[-1] + 1, upper])

    print(missing)
    result = []
    for item in missing:
        if item[0] == item[1]:
            result.append(str(item[0]))
        else:
            out = str(item[0])+"->"+str(item[1])
            result.append(out)

    return result

print(findMissingRanges([-3,10], -2, -1))


def findReplaceString(s, indices, sources, targets):

    def replace_str(start_index, end_index, word, source, replacement_string):
        output = ''
        if source == word[start_index:end_index]:
            word = list(word)
            word[start_index:end_index] = replacement_string
            for l in word:
                output += l
            return output
        else:
            return word

    # Sort the indices
    n = len(indices)
    for i in range(n-1):
        for j in range(0, n-i-1):
            if indices[j] > indices[j+1]:
                indices[j], indices[j+1] = indices[j+1], indices[j]
                sources[j], sources[j+1] = sources[j+1], sources[j]
                targets[j], targets[j+1] = targets[j+1], targets[j]

    word = s
    start_index = indices[0]
    end_index = start_index + len(sources[0])
    word = replace_str(start_index, end_index, word, sources[0], targets[0])
    print(word)


    if len(indices) > 1:
        for i in range(1, len(indices)):
            if len(word) > len(s):
                start_index = indices[i] + (len(word) - len(s))
            else:
                start_index = indices[i] - (len(s) - len(word))
            end_index = start_index + len(sources[i])
            word = replace_str(start_index, end_index, word, sources[i], targets[i])
            print(word)
    return word

print(findReplaceString("abcd", [0, 2, 3], ["x", "c", "d"], ["e", "xxx", "zzz"]))


def backspaceCompare(s,t):
    def remove_chars(word):
        word = list(word)
        i = 1
        while i < len(word):
            if len(word) >= 2:
                if word[i] == '#' and word[i - 1] != '#':
                    del word[i]
                    del word[i - 1]
                    i = 1
                elif word[i-1] == '#':
                    del word[i-1]
                    i = 1
                else:
                    i += 1
            elif word[i] == '#':
                del word[i]
                break
            else:
                break
        return word

    first_output = remove_chars(s)
    second_output = remove_chars(t)

    if first_output == second_output:
        return True
    return False

print(backspaceCompare("ab##","c#d#"))


def letterCombinations(digits):
    # Time: O(n)
    # Space: O(n)
    n = len(digits)
    if not n:
        return []
    alpha = ["0", "1", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"]

    res = ['']

    for i in range(n):
        res = [pre + char for pre in res for char in alpha[int(digits[i])]]

    return res

print(letterCombinations('456'))

def isValidParen(parens):
    stck = list()
    for paren in parens:
        if paren == ')' and stck:
            if stck.pop() != '(':
                return False
        elif paren == '(':
            stck.append(paren)
        else:
            return False

    return len(stck) == 0

def mergeTwoLists(l1, l2):
    if not (l2 or l1): return None
    if not l2: return l1
    if not l1: return l2

    head = l1
    val1 = []
    while head is not None:
        val1.append(head.val)
        head = head.next
    head = l2
    val2 = []
    while head is not None:
        val2.append(head.val)
        head = head.next

    value_list = val1 + val2
    value_list = value_list.sort()
    out = []
    for i in value_list:
        out.append(ListNode(i))

    for i in range(len(out)-1):
        out[i].next = out[i+1]

    return(out[0])




def slices(series, length):
    if length == len(series):
        return[series]
    if length > len(series):
        raise ValueError("slice length cannot be greater than series length")
    if length == 0:
        raise ValueError("slice length cannot be zero")
    if length < 0:
        raise ValueError("slice length cannot be negative")
    if len(series) == 0:
        raise ValueError("series cannot be empty")

    output = []
    start = 0
    end = int(length)
    while end <= len(series):
        output.append(series[start:end])
        start += 1
        end +=1

    return output
print(slices('918493904243', 5))


def find_anagrams(word, candidates):
    output = []
    for candidate in candidates:
        c = candidate.lower()
        word = word.lower()
        if c != word:
            list_word = list(word)
            cand = list(c)
            while len(cand) > 0:
                if cand[0] in list_word:
                    list_word.remove(cand[0])
                    cand.remove(cand[0])
                else:
                    break
            if len(list_word) == 0 and len(cand) == 0:
                output.append(candidate)

    return output

print(find_anagrams("BANANA", ["BANANA", "Banana", "banana"]))

def add_subtract(num_list):
    if len(num_list) <= 1:
        return num_list[0]
    output = num_list[0]
    addition = 1
    for i in range(1, len(num_list)):
        if addition:
            output += num_list[i]
            addition = 0
        else:
            output -= num_list[i]
            addition = 1

    return output

print(add_subtract([7]))


opening_brackets = ['(', '[', '{']
closing_brackets = [')', ']', '}']
bracket_matches = dict(zip(closing_brackets, opening_brackets))
def is_paired(input_string):
    stack = []
    for ch in input_string:
        if ch in opening_brackets:
            stack.append(ch)
        elif ch in closing_brackets:
            if not stack:
                return False
            if bracket_matches[ch] != stack.pop():
                return False
    return len(stack) == 0

print(is_paired("[[]]{}()(((())))[{}]"))

def find_duplicates(arr1, arr2):
    duplicates = []
    i = 0
    j = 0
    while i < len(arr1) and j < len(arr2):
        if arr1[i] == arr2[j]:
            duplicates.append(arr1[i])
            i += 1
            j += 1
        elif arr1[i] < arr2[j]:
            i += 1
        else:
            j += 1

    return duplicates

print(find_duplicates([1, 2, 3, 5, 6, 7], [3, 6, 7, 8, 20]))

def dfs(node):
    keys = []
    def search_tree(root):
        if root:
            keys.append(root.val)
            dfs(root.left)
            dfs(root.right)
    search_tree(node)
    return keys

def getDifferentNumber(arr):
    if 0 not in arr:
        return 0
    arr.sort()
    next_num = 1
    previous_num = 0
    for i in range(1, len(arr)):
        if previous_num == arr[i]:
            continue
        elif next_num != arr[i]:
            return next_num
        else:
            next_num +=1
            previous_num += 1
    return arr[-1]+1

print(getDifferentNumber([0, 2, 1, 3, 3, 3, 5, 5, 5]))

def add_one(given_array):
    # For the given array add 1 to the number
    # [2,3,4] -> [2,3,5]
    # [9,9,9] -> [1,0,0,0]

    # If the given array is empty
    if given_array == []:
        return []

    # Create a new array with the same length
    new_array = [0 for i in range(len(given_array))]

    # Loop through the array and add 1 to it with carry set to 1
    carry = 1
    for i in range(len(given_array)-1, -1, -1):
        if carry:
            sum_out = given_array[i] + 1
            if sum_out == 10:
                carry = 1
                new_array[i] = sum_out % 10
            else:
                carry = 0
                new_array[i] = sum_out
        else:
            new_array[i] = given_array[i]

    if carry == 1:
        result_array = [0 for i in range(len(given_array)+1)]
        result_array[0] = 1
        return result_array
    else:
        return new_array

print(add_one([]))


def rotate(matrix):
    """
    Do not return anything, modify matrix in-place instead.
    """

    def transpose(matrix):
        for i in range(len(matrix)):
            for j in range(i + 1, len(matrix)):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

    def reflect(matrix):
        n = len(matrix)
        m = n //2
        for i in range(n):
            print(i)
            for j in range(m):
                print(-j-1)
                matrix[i][j], matrix[i][-j - 1] = matrix[i][-j - 1], matrix[i][j]

    transpose(matrix)
    reflect(matrix)
    return matrix

print(rotate([[1,2,3,8],[4,5,6,2],[7,8,9,5]]))

import re
def most_common_words(text):
    text = text.strip()
    text = text.split(" ")
    hash_db = {}
    for word in text:
        word = word.lower()
        word = ''.join(re.findall(r'[a-z]', word))
        if word in hash_db:
            hash_db[word] += 1
        else:
            hash_db[word] = 1
    return hash_db

print(most_common_words('It was the best of times, it was the worst of times.'))

from collections import OrderedDict


class LRUCache(OrderedDict):

    def __init__(self, capacity):
        """
        :type capacity: int
        """
        self.capacity = capacity

    def get(self, key):
        """
        :type key: int
        :rtype: int
        """
        if key not in self:
            return - 1

        self.move_to_end(key)
        return self[key]

    def put(self, key, value):
        """
        :type key: int
        :type value: int
        :rtype: void
        """
        if key in self:
            self.move_to_end(key)
        self[key] = value
        if len(self) > self.capacity:
            self.popitem(last=False)

# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)

from heapq import heappush, heappop

def find_largest(input, m):
  max_nums = [0]

  for i in input:
    if i > max_nums[0]:
      if len(max_nums) >= m:
        heappop(max_nums)
      heappush(max_nums, i)

  return max_nums

print(find_largest([1,5,4,2,3], 3))

def get_height(node):
    if not node:
        return 0
    l_height = get_height(node.left)
    r_height = get_height(node.right)

    return max(l_height, r_height) + 1

def is_balanced(node):
    if not node:
        return True
    l_height = get_height(node.left)
    r_height = get_height(node.right)
    diff = l_height - r_height
    if diff > 1:
        return False
    return(is_balanced(node.left) & is_balanced(node.right))

def find_first(array, num):
    start = 0
    end = len(array) -1
    idx = -1
    while start < end:
        mid = int(start + (end - start) / 2)
        if array[mid] == num:
            idx = mid
            end = mid
            continue
        if array[mid] > num:
            end = mid
            continue
        if array[mid] < num:
            start = mid + 1
            continue

    return idx

print(find_first([200, 200, 200, 200, 500, 500, 500], 500))

def find_primes(n):
    # Find all the prime numbers less than or equal to n

    def isNumberPrime(n):
        list_of_divisors = [2,3,4,5]
        for i in list_of_divisors:
            if n == i:
                return 1
            if n % i == 0:
                return 0
        return 1

    result = []
    for i in range(2, n):
        if isNumberPrime(i):
            result.append(i)

    return result

print(find_primes(0))

def max_profit(prices):
    min_result = ''
    max_result = ''
    for place , price in prices.items():
        if min_result != '':
            if prices[min_result] > price:
                min_result = place
        else:
            min_result = place
        if max_result != '':
            if prices[max_result] < price:
                max_result = place
        else:
            max_result = place

    return [min_result, max_result]
print(max_profit({}))

def fib(n):
    if n == 0 or n == 1:
        return 1
    result = [1, 1]
    for i in range(1,n):
        result.append(result[i-1] + result[i])

    return result[-1]
print(fib(5))

def fibRecursive(n):
    if n <= 1:
        return 1
    return fibRecursive(n-1) + fibRecursive(n-2)

print(fibRecursive(5))
