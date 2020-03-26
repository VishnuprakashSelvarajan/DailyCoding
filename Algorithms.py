# coding=utf-8
# Anagram letters
def anagram(s1, s2):
    # Remove spaces and convert the string to lower
    s1 = s1.replace(' ', '').lower()
    s2 = s2.replace(' ', '').lower()

    # compare the string len if not equal return False
    if len(s1) != len(s2): return False

    count = {}  # Count the frequency of letters

    # Count and add the letters in count
    for letter in s1:
        if letter in count:
            count[letter] += 1
        else:
            count[letter] = 1
    # Do the reverse for S2 and remove from the count
    for letter in s2:
        if letter in count:
            count[letter] -= 1
        else:
            count[letter] = 1

    for k in count:
        if count[k] != 0:
            return False

    return True


print(anagram('Clint Eastwood', 'Old West action'))

##############################
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
        return ('Too Small')

    seen = set()
    output = set()
    for num in array:
        target = k - num
        if target not in seen:
            seen.add(num)
        else:
            output.add((min(num, target), max(num, target)))

    return output


print(pair_sum([1, 3, 2, 0, 4, 5, -1, -1, 2, 1, 5, 3, 1, 1, 2, 3], 4))
#####################

'''
Take an array with positive and negative integers and 
find the maximum sum of that array
'''


def largest(arr):
    if len(arr) == 0:
        return False
    max_sum = current_sum = arr[0]
    for num in arr[1:]:
        current_sum = max(current_sum + num, num)
        max_sum = max(current_sum, max_sum)

    return (max_sum)


print(largest([2, 3, -4, 2, 32, 34, -23, 34, 23]))
######################

'''
Given a string of words, reverse all the words
start = "This is the best"
finish = "best the is This
'''


def reverse(s):
    return " ".join(reversed(s.split()))


print(reverse('This is the best'))


def rev(s):
    s = s.split()
    s.reverse()
    return (s)


print(rev('This is the best'))


def reverse(s):
    length = len(s)
    i = 0
    spaces = [' ']
    words = []
    while i < length:
        if s[i] is not spaces:
            word_start = i

            while i < length and s[i] not in spaces:
                i += 1

            words.append(s[word_start:i])
        i += 1

    return " ".join(reversed(s))


print(reverse('This is the best'))


def reverse(s):
    return s[::-1]


print(reverse('This is the best'))

############################
'''
Find the common element in given 2 sorted lists
'''


def common(list_a, list_b):
    p1 = p2 = 0
    result = []
    while p1 < len(list_a) and p2 < len(list_b):
        if list_a[p1] == list_b[p2]:
            result.append(list_a[p1])
            p1 += 1
            p2 += 1
        elif list_a[p1] > list_b[p2]:
            p2 += 1
        else:
            p1 += 1

    return result


print(common([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 9]))

'''
Given an array what is the most frequently occurring element
'''


def most_frequently(list):
    count = {}
    max_count = 0
    max_item = None

    for i in list:
        if i not in count:
            count[i] = 1
        else:
            count[i] += 1
        if count[i] > max_count:
            max_count = count[i]
            max_item = i
    return max_item


print(most_frequently([1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 1, 1, 1]))

'''
Given a string, are all characters unique ?
Should give a True or False return
'''


def unique(string):
    string = string.replace(' ', '')
    characters = set()
    for letter in string:
        if letter in characters:
            return False
        else:
            characters.add(letter)
    return True


print(unique('i jkl reqwi'))

'''
find all the unique letters in a string
'''


def non_repeating(s):
    s = s.replace(' ', '').lower()
    char_count = {}
    for c in s:
        if c in char_count:
            char_count[c] += 1
        else:
            char_count[c] = 1
    unique_list = []
    for c in s:
        if char_count[c] == 1:
            unique_list.append(c)
    return unique_list


print(non_repeating('I apple Ape Peels'))

'''
Given an array A of integers and integer K, 
return the maximum S such that there exists i < j with A[i] + A[j] = S
and S < K. If no i, j exist satisfying this equation, return -1.

Input: A = [34,23,1,24,75,33,54,8], K = 60
Output: 58
Explanation: 
We can use 34 and 24 to sum 58 which is less than 60.

Example 2:

Input: A = [10,20,30], K = 15
Output: -1
'''


def maxsum(array, k):
    max_sum = -1
    ind = 0
    while ind < len(array) - 1:
        target_sum = array[ind] + array[ind + 1]
        if target_sum < k and max_sum < target_sum:
            max_sum = target_sum
        else:
            ind += 1

    return (max_sum)


print(maxsum([34, 23, 1, 24, 75, 33, 54, 8], 200))

'''
You have N gardens, labelled 1 to N.  In each garden, you want to plant one of 4 types of flowers.

paths[i] = [x, y] describes the existence of a bidirectional path from garden x to garden y.

Also, there is no garden that has more than 3 paths coming into or leaving it.

Your task is to choose a flower type for each garden such that, for any two gardens connected by a path, 
they have different types of flowers.

Return any such a choice as an array answer, where answer[i] is the type of flower planted in the (i+1)-th garden.  
The flower types are denoted 1, 2, 3, or 4.  It is guaranteed an answer exists.

Example 1:

Input: N = 3, paths = [[1,2],[2,3],[3,1]]
Output: [1,2,3]

Example 2:

Input: N = 4, paths = [[1,2],[3,4]]
Output: [1,2,1,2]
Example 3:

Input: N = 4, paths = [[1,2],[2,3],[3,4],[4,1],[1,3],[2,4]]
Output: [1,2,3,4]

Note:

1 <= N <= 10000
0 <= paths.size <= 20000
No garden has 4 or more paths coming into or leaving it.
It is guaranteed an answer exists.
'''

'''
First Duplicate
'''


def first_duplicate(array):
    if len(array) < 2:
        return ('-1')
    seen = set()
    for number in array:
        if number in seen:
            return (number)
        else:
            seen.add(number)
    return ('-1')


print(first_duplicate([1, 2, 3, 4, 3, 5, 1, 2]))

'''
num_ways
'''


def num_ways(n):
    if n == 0 or n == 1:
        return 1
    else:
        return num_ways(n - 1) + num_ways(n - 2)


print(num_ways(4))


def nums_bottom_up(n):
    if n == 0 or n == 1:  return 1
    nums = [0 for i in range(n)]

    nums[0] = nums[1] = 1
    for i in range(2, n):
        nums[i] = nums[i - 1] + nums[i - 2]

    return (nums)


print(nums_bottom_up(10))

'''
Read Inorder traversal Binary Tree
'''

'''
Given a list of numbers and a number k, return whether any two numbers from the list add up to k.

For example, given [10, 15, 3, 7] and k of 17, return true since 10 + 7 is 17.
'''


def sumup(array, k):
    seen = set()
    output = []
    if len(array) < 2:
        return False
    for num in array:
        target = k - num
        if target not in seen:
            seen.add(num)
        else:
            output = (min(num, target), max(num, target))
            return True
    return False


print(sumup([10, 15, 3, 7, 10, 7], 17))

'''
You have an array of logs.  Each log is a space delimited string of words.

For each log, the first word in each log is an alphanumeric identifier.  Then, either:

    Each word after the identifier will consist only of lowercase letters, or;
    Each word after the identifier will consist only of digits.

We will call these two varieties of logs letter-logs and digit-logs.  
It is guaranteed that each log has at least one word after its identifier.

Reorder the logs so that all of the letter-logs come before any digit-log.
The letter-logs are ordered lexicographically ignoring identifier, with the identifier used in case of ties.  
The digit-logs should be put in their original order.

Return the final order of the logs.

Example 1:

Input: logs = ["dig1 8 1 5 1","let1 art can","dig2 3 6","let2 own kit dig","let3 art zero"]
Output: ["let1 art can","let3 art zero","let2 own kit dig","dig1 8 1 5 1","dig2 3 6"]

Constraints:

    0 <= logs.length <= 100
    3 <= logs[i].length <= 100
    logs[i] is guaranteed to have an identifier, and a word after the identifier.

'''

'''
Say you have an array for which the ith element is the price of a given stock on day i.

Design an algorithm to find the maximum profit. You may complete as many transactions as you like (i.e., buy one and sell one share of the stock multiple times).

Note: You may not engage in multiple transactions at the same time (i.e., you must sell the stock before you buy again).

Example 1:

Input: [7,1,5,3,6,4]
Output: 7
Explanation: Buy on day 2 (price = 1) and sell on day 3 (price = 5), profit = 5-1 = 4.
             Then buy on day 4 (price = 3) and sell on day 5 (price = 6), profit = 6-3 = 3.

Example 2:

Input: [1,2,3,4,5]
Output: 4
Explanation: Buy on day 1 (price = 1) and sell on day 5 (price = 5), profit = 5-1 = 4.
             Note that you cannot buy on day 1, buy on day 2 and sell them later, as you are
             engaging multiple transactions at the same time. You must sell before buying again.

Example 3:

Input: [7,6,4,3,1]
Output: 0
Explanation: In this case, no transaction is done, i.e. max profit = 0.
'''


def sellmulitple(array):
    maxprofit = 0
    buy_day = 0
    sell_day = buy_day + 1
    for i in range(len(array) - 1):
        max_high = max(array[sell_day:])
        max_high_day = array.index(max_high)
        diff_days = len(array) - max_high_day
        tempprofit = array[max_high_day] - array[buy_day]
        if tempprofit > maxprofit:
            maxprofit = tempprofit
            if diff_days > 3:
                for day in range(len(array[max_high_day:])):
                    next_day = day + 1
                    temp_profit = array[next_day] - array[day]
                    maxprofit = maxprofit + temp_profit

    return (maxprofit)


print(sellmulitple([7, 1, 5, 3, 6, 4]))

'''
121. Best Time to Buy and Sell Stock
Easy

Say you have an array for which the ith element is the price of a given stock on day i.

If you were only permitted to complete at most one transaction (i.e., buy one and sell one share of the stock), design an algorithm to find the maximum profit.

Note that you cannot sell a stock before you buy one.

Example 1:

Input: [7,1,5,3,6,4]
Output: 5
Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.
             Not 7-1 = 6, as selling price needs to be larger than buying price.

Example 2:

Input: [7,6,4,3,1]
Output: 0
Explanation: In this case, no transaction is done, i.e. max profit = 0.
'''


def sellshare(array):
    maxprofit = 0
    buy_day = 0
    if len(array) < 2:
        return maxprofit
    while buy_day < len(array) - 1:
        next_day = buy_day + 1
        temp_maxprofit = max(array[next_day:])
        profit = temp_maxprofit - array[buy_day]
        maxprofit = max(maxprofit, profit)
        buy_day += 1

    return maxprofit


print(sellshare([7, 6, 4, 3, 1]))

'''
Given a string s of '(' , ')' and lowercase English characters. 

Your task is to remove the minimum number of parentheses ( '(' or ')', in any positions ) 
so that the resulting parentheses string is valid and return any valid string.

Formally, a parentheses string is valid if and only if:

    It is the empty string, contains only lowercase characters, or
    It can be written as AB (A concatenated with B), where A and B are valid strings, or
    It can be written as (A), where A is a valid string.

Example 1:

Input: s = "lee(t(c)o)de)"
Output: "lee(t(c)o)de"
Explanation: "lee(t(co)de)" , "lee(t(c)ode)" would also be accepted.

Example 2:

Input: s = "a)b(c)d"
Output: "ab(c)d"

Example 3:

Input: s = "))(("
Output: ""
Explanation: An empty string is also valid.

Example 4:

Input: s = "(a(b(c)d)"
Output: "a(b(c)d)"

Constraints:

    1 <= s.length <= 10^5
    s[i] is one of  '(' , ')' and lowercase English letters.
'''


def minRemoveToMakeValid(s):
    index_to_remove = set()
    seen = set()
    for ind, value in enumerate(s):
        if value not in "()":
            continue
        if value == "(":
            seen.add(ind)
        elif not seen:
            index_to_remove.add(ind)
        else:
            seen.pop()

    join_string = ''
    for ind, value in enumerate(s):
        if ind not in index_to_remove:
            join_string = join_string + value

    return join_string


print(minRemoveToMakeValid('a)b(c)d'))

'''
Given a string s and a string t, check if s is subsequence of t.

You may assume that there is only lower case English letters in both s and t. t is potentially a very long (length ~= 500,000) string, and s is a short string (<=100).

A subsequence of a string is a new string which is formed from the original string by deleting some (can be none) of the characters without disturbing the relative positions of the remaining characters. (ie, "ace" is a subsequence of "abcde" while "aec" is not).

Example 1:
s = "abc", t = "ahbgdc"

Return true.

Example 2:
s = "axc", t = "ahbgdc"

Return false.
'''


# 2 pointer solution

def isSubsequence(s, t):
    i, k = 0, 0

    while i < len(t) and k < len(s):
        if s[k] == t[i]:
            k += 1
        i += 1
    print(k)
    print(len(s))
    return k == len(s)


print(isSubsequence('abc', 'ahbgdc'))

'''
A valid parentheses string is either empty (""), "(" + A + ")", or A + B, where A and B are valid parentheses strings, 
and + represents string concatenation.  For example, "", "()", "(())()", and "(()(()))" are all valid parentheses strings.

A valid parentheses string S is primitive if it is nonempty, and there does not exist a way to split it into S = A+B, with A and B nonempty valid parentheses strings.

Given a valid parentheses string S, consider its primitive decomposition: S = P_1 + P_2 + ... + P_k, where P_i are primitive valid parentheses strings.

Return S after removing the outermost parentheses of every primitive string in the primitive decomposition of S.



Example 1:

Input: "(()())(())"
Output: "()()()"
Explanation: 
The input string is "(()())(())", with primitive decomposition "(()())" + "(())".
After removing outer parentheses of each part, this is "()()" + "()" = "()()()".
Example 2:

Input: "(()())(())(()(()))"
Output: "()()()()(())"
Explanation: 
The input string is "(()())(())(()(()))", with primitive decomposition "(()())" + "(())" + "(()(()))".
After removing outer parentheses of each part, this is "()()" + "()" + "()(())" = "()()()()(())".
Example 3:

Input: "()()"
Output: ""
Explanation: 
The input string is "()()", with primitive decomposition "()" + "()".
After removing outer parentheses of each part, this is "" + "" = "".
'''


def removeOuterParentheses(S):
    output = ''
    stack = []
    count = 0
    for string in S:

        if string == '(':
            count += 1
        else:
            count -= 1
        stack.append(string)

        if count == 0:
            output = output + ''.join(stack[1:-1])
            stack = []
            count = 0

    return output


print(removeOuterParentheses("(()())(())"))


def addDigits(num):
    num = list(str(num))
    sum = 0
    index = 0
    while index <= len(num):
        sum = sum + int(num[index])
        index += 1
        if index == len(num):
            if len(str(sum)) >= 2:
                num = list(str(sum))
                sum = index = 0
            else:
                return sum


print(addDigits(34545))


def findMaxConsecutiveOnes(nums):
    seen = 0
    max_occurance = 0
    index = 0
    if len(nums) < 2 and nums[0] == 1:
        return 1
    while index < len(nums):
        if nums[index] == 1:
            seen += 1
        else:
            seen = 0
        if max_occurance < seen:
            max_occurance = seen
        index += 1
    return max_occurance


print(findMaxConsecutiveOnes([1, 1]))


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


print
twoSum([2, 2, 7, 7, 11, 15], 9)

'''
Given two strings s and t which consist of only lowercase letters.

String t is generated by random shuffling string s and then add one more letter at a random position.

Find the letter that was added in t.

Example:

Input:
s = "abcd"
t = "abcde"

Output:
e

Explanation:
'e' is the letter that was added.
'''


def findExtraLetter(s, t):
    index = 0
    for str in t:
        if str not in s:
            return str

    return ('No new string')


print(findExtraLetter('abcd', 'abcde'))

'''
You have an array of logs.  Each log is a space delimited string of words.

For each log, the first word in each log is an alphanumeric identifier.  Then, either:

Each word after the identifier will consist only of lowercase letters, or;
Each word after the identifier will consist only of digits.
We will call these two varieties of logs letter-logs and digit-logs.  It is guaranteed that each log has at least one word after its identifier.

Reorder the logs so that all of the letter-logs come before any digit-log.  The letter-logs are ordered lexicographically ignoring identifier, with the identifier used in case of ties.  The digit-logs should be put in their original order.

Return the final order of the logs.



Example 1:

Input: logs = ["dig1 8 1 5 1","let1 art can","dig2 3 6","let2 own kit dig","let3 art zero"]
Output: ["let1 art can","let3 art zero","let2 own kit dig","dig1 8 1 5 1","dig2 3 6"]


Constraints:

0 <= logs.length <= 100
3 <= logs[i].length <= 100
logs[i] is guaranteed to have an identifier, and a word after the identifier.
'''

'''
Longest Palindrom
Given a string which consists of lowercase or uppercase letters, 
find the length of the longest palindromes that can be built with those letters.

This is case sensitive, for example "Aa" is not considered a palindrome here.

Note:
Assume the length of given string will not exceed 1,010.

Example:

Input:
"abccccdd"

Output:
7

Explanation:
One longest palindrome that can be built is "dccaccd", whose length is 7.
'''


def longest_palindrom(s):
    ht = {}
    for char in s:
        if char not in ht:
            ht[char] = 1
            continue
        ht[char] += 1
    count = 0
    odds = []
    for key in ht:
        if ht[key] % 2 == 0:
            count += ht[key]
            print(key)
            continue
        odds.append(ht[key])

    if len(odds) == 0:
        return count
    odds.sort()
    count += odds.pop()
    for i in range(len(odds)):
        odds[i] -= 1
    return count + sum(odds)


from collections import Counter


def longest_palindrom2(s):
    count = sum([(x // 2) * 2 for x in Counter(s).values()])
    return count if count == len(s) else (count + 1)


print(longest_palindrom2('abccccdd'))

'''
Range Addition II
Given an m * n matrix M initialized with all 0's and several update operations.

Operations are represented by a 2D array, and each operation is represented by an array with two positive integers a and b, 
which means M[i][j] should be added by one for all 0 <= i < a and 0 <= j < b.

You need to count and return the number of maximum integers in the matrix after performing all the operations.

Example 1:
Input: 
m = 3, n = 3
operations = [[2,2],[3,3]]
Output: 4
Explanation: 
Initially, M = 
[[0, 0, 0],
 [0, 0, 0],
 [0, 0, 0]]

After performing [2,2], M = 
[[1, 1, 0],
 [1, 1, 0],
 [0, 0, 0]]

After performing [3,3], M = 
[[2, 2, 1],
 [2, 2, 1],
 [1, 1, 1]]

So the maximum integer in M is 2, and there are four of it in M. So return 4.
'''

'''
121.Best time to buy and sell
Say you have an array for which the ith element is the price of a given stock on day i.

If you were only permitted to complete at most one transaction (i.e., buy one and sell one share of the stock), design an algorithm to find the maximum profit.

Note that you cannot sell a stock before you buy one.

Example 1:

Input: [7,1,5,3,6,4]
Output: 5
Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.
             Not 7-1 = 6, as selling price needs to be larger than buying price.
Example 2:

Input: [7,6,4,3,1]
Output: 0
Explanation: In this case, no transaction is done, i.e. max profit = 0.
'''

'''
415. Add Strings
Given two non-negative integers num1 and num2 represented as string, return the sum of num1 and num2.

Note:

The length of both num1 and num2 is < 5100.
Both num1 and num2 contains only digits 0-9.
Both num1 and num2 does not contain any leading zero.
You must not use any built-in BigInteger library or convert the inputs to integer directly.
'''

'''
720 Longest Word in Dictionary

Given a list of strings words representing an English Dictionary, 
find the longest word in words that can be built one character at a time by other words in words. 
If there is more than one possible answer, return the longest word with the smallest lexicographical order.

If there is no answer, return the empty string.
Example 1:
Input: 
words = ["w","wo","wor","worl", "world"]
Output: "world"
Explanation: 
The word "world" can be built one character at a time by "w", "wo", "wor", and "worl".
Example 2:
Input: 
words = ["a", "banana", "app", "appl", "ap", "apply", "apple"]
Output: "apple"
Explanation: 
Both "apply" and "apple" can be built from other words in the dictionary. However, "apple" is lexicographically smaller than "apply".
'''

'''
This problem was asked by Uber.

Given an array of integers, return a new array such that each element at index i of the new array is the product of 
all the numbers in the original array except the one at i.

For example, if our input was [1, 2, 3, 4, 5], the expected output would be [120, 60, 40, 30, 24]. If our input was [3, 2, 1], 
the expected output would be [2, 3, 6].

Follow-up: what if you can't use division?

'''


def product_array(input_array):
    output_array = []
    output_array = [1 for i in range(len(input_array))]
    for i in range(len(input_array)):
        output_array.append(1)

    for i in range(len(input_array)):
        pop_value = input_array.pop(i)
        output_array[i] = reduce(lambda x, y: x * y, input_array)
        input_array.insert(i, pop_value)

    return output_array


print(product_array([1, 2, 3, 4, 5]))

'''
Given two arrays, write a function to compute their intersection.

Example 1:

Input: nums1 = [1,2,2,1], nums2 = [2,2]
Output: [2,2]
Example 2:

Input: nums1 = [4,9,5], nums2 = [9,4,9,8,4]
Output: [4,9]
Note:

Each element in the result should appear as many times as it shows in both arrays.
The result can be in any order.
Follow up:

What if the given array is already sorted? How would you optimize your algorithm?
What if nums1's size is small compared to nums2's size? Which algorithm is better?
What if elements of nums2 are stored on disk, and the memory is limited such that you cannot load all elements into the memory at once?
'''


def intersection(nums1, nums2):
    seen = []
    if len(nums1) < len(nums2):
        short_len_arr = nums1
        long_len_arr = nums2
    elif len(nums1) > len(nums2):
        short_len_arr = nums2
        long_len_arr = nums1
    else:
        short_len_arr = nums2
        long_len_arr = nums1

    num1 = num2 = 0
    while num1 < len(short_len_arr):
        if short_len_arr[num1] in long_len_arr:
            seen.append(short_len_arr[num1])
            long_len_arr.remove(short_len_arr[num1])
        num1 += 1

    return (seen)


def commonChars(A):
    match_str = A[0]
    output = []
    i = 0
    for strng in match_str:
        count = match_str.count(strng)
        occurances = 1
        for index in range(1, len(A)):
            if strng in A[index]:
                count = min(count, A[index].count(strng))
                occurances += 1
            else:
                break
        if occurances == len(A):
            for i in range(count):
                output.append(strng)

    return output


print(commonChars(["bella", "label", "roller"]))


def majorityElement(nums):
    count = 0
    output = 0
    for num in nums:
        if count == 0:
            output = num
        if output == num:
            count += 1
        else:
            count -= 1
    return output


print(majorityElement([3, 1, 1, 1, 1, 2, 2, 2, 1, 1]))

'''
Given an array of integers, 
find the first missing positive integer in linear time and constant space. 
In other words, find the lowest positive integer that does not exist in the array. 
The array can contain duplicates and negative numbers as well.

For example, the input [3, 4, -1, 1] should give 2. The input [1, 2, 0] should give 3.

You can modify the input array in-place.
'''


def missingpositive(array):
    sorted_array = sorted(array)
    i = 0
    last_index = len(array) - 1
    positive = sorted_array[last_index] + 1
    while i < len(array) and last_index > i:
        target = sorted_array[i] + sorted_array[last_index]
        if target not in array:
            if positive > target:
                positive = target
                i += 1
        else:
            last_index -= 1
    return positive


print(missingpositive([1, 2, 0, 3, 1, 2, -1, 5]))

'''
Merge N sorted lists to a new sorted list (Heap - merging multiple sorted lists to a single sorted list)
'''


def merge(lists):
    merged_list = []
    import heapq
    heap = [(lst[0], i, 0) for i, lst in enumerate(lists) if lst]
    heapq.heapify(heap)

    while heap:
        val, list_ind, element_ind = heapq.heappop(heap)

        merged_list.append(val)

        if element_ind + 1 < len(lists[list_ind]):
            next_tuple = (lists[list_ind][element_ind + 1],
                          list_ind,
                          element_ind + 1)
            heapq.heappush(heap, next_tuple)
    return merged_list


print(merge([[12, 15, 20], [17, 20, 32], [10, 15, 30]]))

'''
Fibnocci Series
'''


def fibnocci(lgth):
    output = [0]
    a, b = 0, 1
    for i in range(lgth):
        a, b = b, a + b
        output.append(a)
    return output


print(fibnocci(10))


def isHappy(n):
    if n < 0:
        return False

    seen = set()
    while n:
        if n == 1:
            return True
        if n in seen:
            return False
        else:
            seen.add(n)
        if n < 4:
            return False
        else:
            n = sum(map(lambda x: int(x) ** 2, list(str(n))))


print(isHappy(19))

'''
Binary Search
'''


def binary_search(data, target):
    low = 0
    high = len(data) - 1
    while low <= high:

        mid = (low + high) // 2

        if target == data[mid]:
            return True
        elif target < data[mid]:
            high = mid - 1
        else:
            low = mid + 1

    return False


print(binary_search([1, 2, 45, 67, 78, 90, 101, 234, 23523], 101))

'''
Given the mapping a = 1, b = 2, ... z = 26, and an encoded message, count the number of ways it can be decoded.

For example, the message '111' would give 3, since it could be decoded as 'aaa', 'ka', and 'ak'.

You can assume that the messages are decodable. For example, '001' is not allowed.
'''


def helper(data, k):
    if k == 0:
        return 1
    s = len(data) - k  # This is to get the first Kth element of the data
    if data[s] == '0':  # This is the return 0 if there is 0 in the starting of the data
        return 0

    result = helper(data, k - 1)  # This is  for the first number and decoder of the rest of the data
    if k >= 2 and int(data[s:s + 2]) < 27:
        result += helper(data, k - 2)

    return result


def num_ways(data):
    k = len(data)
    return (helper(data, k))


print(num_ways('1111'))


######

def helper_dp(data, k, memo):
    if k == 0:
        return 0
    s = len(data) - k
    if data[s] == '0':
        return 0
    if memo[k] != 'null':
        return memo[k]
    result = helper_dp(data, k - 1, memo)
    if k > 2 and int(data[s:s + 2]) > 27:
        result += helper_dp(data, k - 2, memo)
    memo[k] = result


def num_ways_dp(data):
    memo = ['null' for i in range(len(data) + 1)]
    return (helper_dp(data, len(data), memo))


print(num_ways_dp('111'))

'''
All Subsets of a set
'''


def all_subsets(array):
    subset = ['null' for i in range(len(array))]
    helper_subset(array, subset, 0)


def helper_subset(array, subset, i):
    if i == len(array):
        print(subset)
    else:
        subset[i] = 'null'
        helper_subset(array, subset, i + 1)
        subset[i] = array[i]
        helper_subset(array, subset, i + 1)


print(all_subsets([1, 2]))


def subsets_binary_way(array):
    for i in range(pow(2, len(array))):
        binary_array = [int(j) for j in bin(i)[2:]]

        zero_array = [0] * (len(array) - len(binary_array))

        binary_array_full = zero_array + binary_array
        sub_array = [array[k] * binary_array_full[k] for k in range(len(binary_array_full))]

        print([i for i in sub_array if i != 0])


print(subsets_binary_way([1, 2, 3, 4]))

'''
All Subsets of a set
'''


def subs(l):
    if l == []:
        return [[]]

    x = subs(l[1:])
    print('X Value is', x)
    return x + [[l[0]] + y for y in x]


print(subs([1, 2, 3, 4]))

'''
Write a function that receives an array 'arr' of processable non-unique integers and a cooldown 'c'.
This function returns the total execution time for the array

Each unique integer has a cooldown of 'c' seconds before another integer of the same value can be processed.
Each integer requires 1 sec for processing

Eg: 
input: [1,2,1,1,2,2,2], 2
output: 14
'''


def getTotalExecutionTime(arr, c):
    exec_time = 0
    if len(arr) < 2:
        return 1
    seen = set()
    index = 0
    while index < len(arr):
        if arr[index] in seen:
            exec_time += 2
        else:
            seen.add(arr[index])
            exec_time += 1
        index += 1
    return exec_time


print(getTotalExecutionTime([1, 2, 1, 1, 2, 2, 2], 2))

'''
n a deck of cards, every card has a unique integer.  You can order the deck in any order you want.

Initially, all the cards start face down (unrevealed) in one deck.

Now, you do the following steps repeatedly, until all cards are revealed:

Take the top card of the deck, reveal it, and take it out of the deck.
If there are still cards in the deck, put the next top card of the deck at the bottom of the deck.
If there are still unrevealed cards, go back to step 1.  Otherwise, stop.
Return an ordering of the deck that would reveal the cards in increasing order.

The first entry in the answer is considered to be the top of the deck.



Example 1:

Input: [17,13,11,2,3,5,7]
Output: [2,13,3,11,5,17,7]
Explanation: 
We get the deck in the order [17,13,11,2,3,5,7] (this order doesn't matter), and reorder it.
After reordering, the deck starts as [2,13,3,11,5,17,7], where 2 is the top of the deck.
We reveal 2, and move 13 to the bottom.  The deck is now [3,11,5,17,7,13].
We reveal 3, and move 11 to the bottom.  The deck is now [5,17,7,13,11].
We reveal 5, and move 17 to the bottom.  The deck is now [7,13,11,17].
We reveal 7, and move 13 to the bottom.  The deck is now [11,17,13].
We reveal 11, and move 17 to the bottom.  The deck is now [13,17].
We reveal 13, and move 17 to the bottom.  The deck is now [17].
We reveal 17.
Since all the cards revealed are in increasing order, the answer is correct.
'''


def deckRevealedIncreasing(deck):
    d = {}
    l = [x for x in range(len(deck))]
    l1 = []
    deck = sorted(deck)
    while len(l) > 1:
        l1.append(l.pop(0))
        l.append(l.pop(0))
    l1 += l
    for i in range(len(deck)):
        d[l1[i]] = deck[i]
    for i in d.keys():
        l1[i] = d[i]
    return (l1)


print(deckRevealedIncreasing([17, 13, 11, 2, 3, 5, 7]))


def deckReveal(deck):
    if len(deck) < 3:
        return (sorted(deck))
    print(deck)
    deck = sorted(deck, reverse=True)
    ordered_deck = []
    ordered_deck.append(deck[1])
    ordered_deck.append(deck[0])

    for index in range(2, len(deck)):
        print(ordered_deck)
        ordered_deck.insert(0, ordered_deck[len(ordered_deck) - 1])
        ordered_deck.pop(len(ordered_deck) - 1)
        ordered_deck.insert(0, deck[index])

    return (ordered_deck)


'''
Given an array A of integers, return true if and only if we can partition the array into three non-empty parts with equal sums.

Formally, we can partition the array if we can find indexes i+1 < j with (A[0] + A[1] + ... + A[i] == A[i+1] + A[i+2] + ... + A[j-1] == A[j] + A[j-1] + ... + A[A.length - 1])



Example 1:

Input: [0,2,1,-6,6,-7,9,1,2,0,1]
Output: true
Explanation: 0 + 2 + 1 = -6 + 6 - 7 + 9 + 1 = 2 + 0 + 1
Example 2:

Input: [0,2,1,-6,6,7,9,-1,2,0,1]
Output: false
Example 3:

Input: [3,3,6,5,-2,2,5,1,-9,4]
Output: true
Explanation: 3 + 3 = 6 = 5 - 2 + 2 + 5 + 1 - 9 + 4
'''
'''
We are asked to determine if the array can be partioned into three adjacent regions each of which have the same sum. 
Since we must use all of the elements in the array the sum of each of the three regions must be sum(A)//3. 
If sum(A) is not divisible by 3, one can immediately return False because a number which is not a multiple of 3 cannot be divided evenly into three equal numbers. 
The program starts by setting g equal to the sum that each region must be. The variable C is initialized to zero and will hold the cumulative sum as we iterate from left to right. The variable p is initialized to zero and is the count of the number of successful partition points we have found thus far. Since the iteration is yet to begin, we have not yet found any partition points so p is equal to zero. The goal of the program is to successfully find two partition points or partition indices such that the sum of the values within each region is equal.

The for loop iterates through each element of A from left to right, adding the element to the the cumulative sum C. If the variable C eventually equals g then that means that we've found a continuous group of numbers starting from the beginning of the list that add up to our goal sum for each of the three regions. So if C == g, we can set p equal to one because we have now found the first partition point. We reset the cumulative sum C to zero to being accumulating the numbers in the second region. Note that if C never reaches the goal sum g then there can be no first partition and the for loop will eventually finish and the program will return False.

Now that p equals one and C has been reset to zero we continue on with the loop, adding the elements to C one by one. Note that we are now currently in the middle (or the second) of the three partitions that we hope to find. Once again as C increases (or decreases if the numbers are negative) we check to see if C has reached our goal sum of g yet. Again, if C never reaches the goal sum g then there can be no second partition and the for loop will eventually finish and the program will return False. If we do successfully reach the goal sum g (i.e. C == g) for the second time, we have successfully found the second partition point.

At this point the program can return True and end. The reason that we don't have to search any further or add any more numbers is because if the first two regions each have a sum of g then the third region must also add to g since g was defined to be one third of the total sum of array A.

I've also included a one line version just for fun!

'''


def canThreePartsEqualSum(A):
    S = sum(A)
    if S % 3 != 0: return False
    g, C, p = S // 3, 0, 0
    for a in A:
        C += a
        if C == g:
            if p == 1: return True
            C, p = 0, 1
    return False


def canThreePartsEqualSum_oneline(A):
    return (lambda x, y: x in y and 2 * x in y)(sum(A) // 3, itertools.accumulate(A))


'''
Given a list of integers, write a function that returns the largest sum of non-adjacent numbers. Numbers can be 0 or negative.

For example, [2, 4, 6, 2, 5] should return 13, since we pick 2, 6, and 5. [5, 1, 1, 5] should return 10, since we pick 5 and 5.

Follow-up: Can you do this in O(N) time and constant space?
'''


def getMaxSumNonAdj(array):
    if len(array) < 2:
        return (array)
    if len(array) == 2:
        return max(array[0], array[1])

    start_index = 0
    target_sum = array[0]
    while start_index < len(array) - 1:
        temp_sum = array[start_index]
        for sub_index in range(start_index + 2, len(array), 2):
            print(array[start_index], array[sub_index])
            temp_sum += array[sub_index]
        target_sum = max(temp_sum, target_sum)
        start_index += 1

    start_index = 0
    while start_index < len(array) - 1:
        for sub_index in range(start_index + 2, len(array)):
            print(array[start_index], array[sub_index])
            target_sum = max((array[start_index] + array[sub_index]), target_sum)
        start_index += 1

    return target_sum


print(getMaxSumNonAdj([2, 4, 6, 2, 5]))

'''
You have a list of words and a pattern, and you want to know which words in words matches the pattern.

A word matches the pattern if there exists a permutation of letters p so that after replacing every letter x in the pattern with p(x), 
we get the desired word.

(Recall that a permutation of letters is a bijection from letters to letters: every letter maps to another letter, and no two letters map to the same letter.)

Return a list of the words in words that match the given pattern. 

You may return the answer in any order.

Example 1:

Input: words = ["abc","deq","mee","aqq","dkd","ccc"], pattern = "abb"
Output: ["mee","aqq"]
Explanation: "mee" matches the pattern because there is a permutation {a -> m, b -> e, ...}. 
"ccc" does not match the pattern because {a -> c, b -> c, ...} is not a permutation,
since a and b map to the same letter.
'''


def findAndReplacePattern(words, pattern):
    pattern_hash = {}
    count = 1
    for letter in pattern:
        if letter in pattern_hash:
            continue
        else:
            pattern_hash[letter] = count
        count += 1

    pattern_val = []
    for letter in pattern:
        pattern_val.append(pattern_hash[letter])

    output_pattern = []

    for word in words:
        current_pattern = {}
        count = 1
        for letter in word:
            if letter in current_pattern:
                continue
            else:
                current_pattern[letter] = count
            count += 1

        current_value = []
        for letter in word:
            current_value.append(current_pattern[letter])

        if current_value == pattern_val:
            output_pattern.append(word)

    return output_pattern


print(findAndReplacePattern(["abc", "deq", "mee", "aqq", "dkd", "ccc"], "abb"))

'''
Given an array of integers arr and two integers k and threshold.

Return the number of sub-arrays of size k and average greater than or equal to threshold.



Example 1:

Input: arr = [2,2,2,2,5,5,5,8], k = 3, threshold = 4
Output: 3
Explanation: Sub-arrays [2,5,5],[5,5,5] and [5,5,8] have averages 4, 5 and 6 respectively. All other sub-arrays of size 3 have averages less than 4 (the threshold).
Example 2:

Input: arr = [1,1,1,1,1], k = 1, threshold = 0
Output: 5
Example 3:

Input: arr = [11,13,17,23,29,31,7,5,2,3], k = 3, threshold = 5
Output: 6
Explanation: The first 6 sub-arrays of size 3 have averages greater than 5. Note that averages are not integers.
Example 4:

Input: arr = [7,7,7,7,7,7,7], k = 7, threshold = 7
Output: 1
Example 5:

Input: arr = [4,4,4,4], k = 4, threshold = 1
Output: 1
'''


def numOfSubarrays(arr, k, threshold):
    count, subArr, total = 0, [], 0

    for n in arr:
        subArr.append(n)
        if len(subArr) > k:
            total -= subArr.pop(0)
        total += n
        if len(subArr) == k and total / k >= threshold:
            count += 1

    return count


print(numOfSubarrays([2, 2, 2, 2, 5, 5, 5, 8], 3, 5))


def findBestValue(arr, target):
    import sys
    min_value = target // len(arr)
    max_value = max(arr)
    min_value_key = min_value - 1
    diffs = {min_value_key: sys.maxsize}
    arr.sort()

    for value in range(min_value, max_value + 1):
        count = 0
        for i in range(len(arr)):
            if arr[i] > value:
                count += value
            else:
                count += arr[i]
        diffs[value] = abs(count - target)
        if diffs[value] == 0:
            return value
        if diffs[value - 1] <= diffs[value]:
            return value - 1


print(findBestValue([2, 3, 5], 10))

'''
Binary Tree Traversal

start = root


def preOrderTraversal(start, traversal):
    if start:
        traversal.append(start.val)
        traversal = preOrderTraversal(start.left, traversal)
        traversal = preOrderTraversal(start.right, traversal)

    return traversal


print(preOrderTraversal(start, []))


def postOrderTraversal(start, traversal):
    if start:
        traversal = postOrderTraversal(start.left, traversal)
        traversal = postOrderTraversal(start.right, traversal)
        traversal.append(start.val)

    return traversal


print(postOrderTraversal(start, []))


def inOrderTraversal(start, traversal):
    if start:
        traversal = inOrderTraversal(start.left, traversal)
        traversal.append(start.val)
        traversal = inOrderTraversal(start.right, traversal)

    return traversal


print(inOrderTraversal(start, []))
'''
'''
404. Sum of Left Leaves

Find the sum of all left leaves in a given binary tree.

Example:

    3
   / \
  9  20
    /  \
   15   7

There are two left leaves in the binary tree, with values 9 and 15 respectively. Return 24.
'''


class Solution:
    def sumOfLeftLeaves(self, root):
        if not root:
            return 0
        self.left_sum = []

        def helper(root, left_flg):

            if not root.left and not root.right and left_flg == 1:
                self.left_sum.append(root.val)

            if root.left:
                helper(root.left, 1)
            if root.right:
                helper(root.right, 0)

            return self.left_sum

        helper(root, 0)
        return sum(self.left_sum)


# print(Solution.sumOfLeftLeaves(self, [3,9,20,null,null,15,7]))

'''
Given a string S of '(' and ')' parentheses, we add the minimum number of parentheses ( '(' or ')', and in any positions ) so that the resulting parentheses string is valid.

Formally, a parentheses string is valid if and only if:

It is the empty string, or
It can be written as AB (A concatenated with B), where A and B are valid strings, or
It can be written as (A), where A is a valid string.
Given a parentheses string, return the minimum number of parentheses we must add to make the resulting string valid.



Example 1:

Input: "())"
Output: 1
Example 2:

Input: "((("
Output: 3
Example 3:

Input: "()"
Output: 0
Example 4:

Input: "()))(("
Output: 4
'''


def minAddToMakeValid(S):
    stack, count = [], 0

    for i in S:
        if i == '(':
            stack.append(i)
        elif stack:
            stack.pop()
        else:
            count += 1

    print(stack, count)

    return len(stack) + count


print(minAddToMakeValid("())))(("))


def remove_leading_zero(num):
    i = 0
    while i < len(num) and num[i] == '0':
        i += 1

    if i == len(num):
        return '0'
    return num[i:]


# ITERATIVE SOLUTION
def removeKdigits(num, k):
    if k == len(num):
        return '0'

    for i in range(k):
        to_remove_index = 0
        while to_remove_index < len(num) - 1 and num[to_remove_index] <= num[to_remove_index + 1]:
            to_remove_index += 1
        num = num[:to_remove_index] + num[to_remove_index + 1:]

    return remove_leading_zero(num)


print(removeKdigits("1432219", 3))


def customSortString(S, T):
    S, T = list(S), list(T)
    output = []
    temp_list = T
    index = 0
    for match in S:
        while match in T:
            if T[index] == match:
                output.append(T[index])
                T.pop(index)
                index = 0
            else:
                index += 1

    print(temp_list)
    print(output)

    return output


print(customSortString("cba", "ggagbbbccgcd"))


def queryMatching(s, query):
    for word in s:
        if word[0] == '*':
            word = word.strip('*')
            if word in query:   return True
        elif word[len(word) - 1] == '*':
            word = word.strip('*')
            if word in query[0:len(word)]:  return True
        elif '*' in word:
            star_index = word.index('*')
            if word[0:star_index] in query and word[star_index + 1:len(word)] in query:
                return True
        elif word == query:
            return True

    return False


print(queryMatching(['aa*', 'ac*dd', '*abc'], 'acbd'))


# Facebook logo stickers cost $2 each from the company store. I have an idea.
# I want to cut up the stickers, and use the letters to make other words/phrases.
# A Facebook logo sticker contains only the word 'facebook', in all lower-case letters.
#
# Write a function that, given a string consisting of a word or words made up # of letters from the word 'facebook', outputs an integer with the number of # stickers I will need to buy.
#
# get_num_stickers('coffee kebab') -> 3
# get_num_stickers('book') -> 1
# get_num_stickers('ffacebook') -> 2
#
# You can assume the input you are passed is valid, that is, does not contain # any non-'facebook' letters, and the only potential non-letter characters # in the string are spaces.

# Solution :

def fb_stkr(x):
    map_str = 'facebook'
    s1 = list(map_str)
    x_list = list(x)
    count = 1
    for i in x_list:
        if i in s1:
            s1.remove(i)
        else:
            s1.extend(list(map_str))
            s1.remove(i)
            count += 1

    return (count)


x = 'bookoffeeoeo'
print(fb_stkr(x))


# Longest Prefix Match
#
def long_prefix(s):
    st_idx = 0
    ed_idx = st_idx + 1
    temp = ''
    lg_pfx = ''
    print('Total length of String is ', len(s))
    s = s + '0'
    while ed_idx < len(s):
        print(ed_idx)
        if s[st_idx] == s[ed_idx]:
            st_idx = ed_idx
            ed_idx += 1
            if len(temp) > len(lg_pfx):
                lg_pfx = temp
        elif s[ed_idx] in temp:
            st_idx = ed_idx
            ed_idx += 1
        else:
            print(st_idx, ed_idx)
            ed_idx2 = ed_idx + 1
            temp = s[st_idx:ed_idx2]
            ed_idx += 1
            print(temp)

    return (lg_pfx)


print(long_prefix('abcabcd'))


def containsNearbyDuplicate(nums, k):
    if len(nums) == len(set(nums)):
        # Reject array with distinct element
        return False
    for i, cur_value in enumerate(nums):

        # Sliding window from i+1 to i+k
        for j in range(i + 1, i + k + 1, 1):

            if j >= len(nums):
                continue

            else:
                if nums[j] == cur_value:
                    return True

    return False


'''
Merge Sort
=== Merge Sort ====
# Time Complexity is O(n)
'''


def merge(list_x, lft_list, rgt_list):
    full = len(list_x)
    nl = len(lft_list)
    nr = len(rgt_list)
    i = j = k = 0
    while i < nl and j < nr:
        if lft_list[i] <= rgt_list[j]:
            list_x[k] = lft_list[i]
            i += 1
            k += 1
        else:
            list_x[k] = rgt_list[j]
            j += 1
            k += 1
    while i < nl:
        list_x[k] = lft_list[i]
        i += 1
        k += 1
    while j < nr:
        list_x[k] = rgt_list[j]
        j += 1
        k += 1


def merge_sort(list_x):
    l = len(list_x)
    if l < 2:
        return (list_x)
    lft = int(l / 2)
    lft_list = list_x[0:lft]
    rgt = l - lft
    rgt_list = list_x[lft:l]
    merge_sort(lft_list)
    merge_sort(rgt_list)
    merge(list_x, lft_list, rgt_list)
    return (list_x)


list_a = [12, 34, 21, 43, 6, 9, 3, 45, 23, 54, 34]
print(merge_sort(list_a))

'''
Quick Sort
Worst Case: O(n2)
Best Case: O(n log n)
'''


def quick_sort(A):
    quick_sort2(A, 0, len(A) - 1)


def quick_sort2(A, low, hi):
    if low < hi:
        p = partition(A, low, hi)
        quick_sort2(A, low, p - 1)  # All items on left of pivot
        quick_sort2(A, p + 1, hi)  # All items on right of the pivot


def get_pivot(A, low, hi):
    mid = (hi + low) // 2
    pivot = hi
    if A[low] < A[mid]:
        if A[mid] < A[hi]:
            pivot = mid
    elif A[low] < A[hi]:
        pivot = low
    return pivot


def partition(A, low, hi):
    pivotIndex = get_pivot(A, low, hi)
    pivotValue = A[pivotIndex]
    A[pivotIndex], A[low] = A[low], A[pivotIndex]
    border = low

    for i in range(low, hi + 1):
        if A[i] < pivotValue:
            border += 1
            A[i], A[border] = A[border], A[i]
    A[low], A[border] = A[border], A[low]

    return border


print(quick_sort([2, 7, 4, 6, 34, 2, 34, 6, 3, 2, 3, 23, 2, 46, 23, 3, 4, 3, 2, 3, 3, 3, 35, 5, 5, 6, 6]))


def quick_sort(A):
    quick_sort2(A, 0, len(A) - 1)


def quick_sort2(A, low, hi):
    if hi - low < 10 and low < hi:
        quick_selection(A, low, hi)
    elif low < hi:
        p = partition(A, low, hi)
        quick_sort2(A, low, p - 1)
        quick_sort2(A, p + 1, hi)


def get_pivot(A, low, hi):
    mid = (hi + low) // 2
    s = sorted([A[low], A[mid], A[hi]])
    if s[1] == A[low]:
        return low
    elif s[1] == A[mid]:
        return mid
    return hi


def partition(A, low, hi):
    pivotIndex = get_pivot(A, low, hi)
    pivotValue = A[pivotIndex]
    A[pivotIndex], A[low] = A[low], A[pivotIndex]
    border = low

    for i in range(low, hi + 1):
        if A[i] < pivotValue:
            border += 1
            A[i], A[border] = A[border], A[i]
    A[low], A[border] = A[border], A[low]

    return (border)


def quick_selection(x, first, last):
    for i in range(first, last):
        minIndex = i
        for j in range(i + 1, last + 1):
            if x[j] < x[minIndex]:
                minIndex = j
        if minIndex != i:
            x[i], x[minIndex] = x[minIndex], x[i]


A = [5, 9, 1, 2, 4, 8, 6, 3, 7]
print(A)
quick_sort(A)
print(A)

# Differences between TCP and UDP, find all files that share the same inode number
## find -inum n or find -samefile
'''
This problem was asked by Google.

The area of a circle is defined as πr^2. 
Estimate π to 3 decimal places using a Monte Carlo method.

Hint: The basic equation of a circle is x2 + y2 = r2.
'''

'''
Given a list split into 2 and both the sum of the lists should be same
a = [5,2,3]
output = [5] [2,3]
'''

'''
Generate palindrome pairs
input = ['code', 'edoc', 'da', 'd']
'''


def palindrome_pairs(words):
    result = []
    for i, word1 in enumerate(words):
        for j, word2 in enumerate(words):
            if i != j:
                word = word1 + word2
                if word == word[::-1]:
                    result.append((i, j))

    return result


print(palindrome_pairs(['code', 'edoc', 'da', 'd']))

'''
Detemine whether the brackets are balanced
eg: Input = "([])[]({})"
'''


def balancedBrackets(s):
    seen = []
    for bracket in s:
        if bracket in ["[", "(", "{"]:
            seen.append(bracket)
        else:
            if not seen:
                return False

            if (bracket == ")" and seen[-1] != "(") or (bracket == "]" and seen[-1] != "[") or (
                    bracket == "}" and seen[-1] != "{"):
                return False

            seen.pop()

    return len(seen) == 0


print(balancedBrackets("([])[]({})"))

'''
Find Most similar wesites
'''
from collections import defaultdict
import heapq


def compute_similarity(a, b, visitors):
    return len(visitors[a] & visitors[b]) / len(visitors[a] | visitors[b])


def top_pairs(log, k):
    visitors = defaultdict(set)
    for site, user in log:
        visitors[site].add(user)

    pairs = []
    sites = list(visitors.keys())

    for _ in range(k):
        heapq.heappush(pairs, (0, ('', '')))

    for i in range(len(sites) - 1):
        for j in range(i + 1, len(sites)):
            score = compute_similarity(sites[i], sites[j], visitors)
            heapq.heappushpop(pairs, (score, (sites[i], sites[j])))

    return [pair[1] for pair in pairs]


'''
Print last lines of a file
'''

import sys
import os


def file_read_from_tail(fname, lines):
    bufsize = 8192
    fsize = os.stat(fname).st_size
    iter = 0
    with open(fname) as f:
        if bufsize > fsize:
            bufsize = fsize - 1
            data = []
            while True:
                iter += 1
                f.seek(fsize - bufsize * iter)
                data.extend(f.readlines())
                if len(data) >= lines or f.tell() == 0:
                    print(''.join(data[-lines:]))
                    break


file_read_from_tail('test.txt', 2)

'''Simple but inefficient way'''
with open('file.txt', 'rb') as f:
    f = f.readlines()
    print(f[-2:])























