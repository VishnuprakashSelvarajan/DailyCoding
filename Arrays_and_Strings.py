"""
1. Is Unique
Implement an algorithm to determine if a string has all unique characters.
What if you cannot use additional data structures ?
"""

def is_unique(string):
    seen = {}
    for s in string:
        if s in seen:
            return False
        else:
            seen[s] = 1
    return True
print(is_unique('132'))

"""
2. Check Permutations
Given 2 strings, write a method to decide if one is a permutation of the other
"""
def permutation_check(string1, string2):
    str1_dict = {}
    str2_dict = {}
    for s1, s2 in zip(string1, string2):
        if s1 in str1_dict:
            str1_dict[s1] += 1
        else:
            str1_dict[s1] = 1
        if s2 in str2_dict:
            str2_dict[s2] += 1
        else:
            str2_dict[s2] = 1

    str1_counts = [count for count in str1_dict.values()]
    str2_count2 = [count for count in str2_dict.values()]
    if str1_counts == str2_count2:
        return True
    else:
        return False

print(permutation_check('string1', 'string2'))

"""
3. URLify:
Write a method to replace all spaces in a string with '%20'. You may assume that the string has sufficient
space at the end to hold the additional characters, and that you are given the "true" length of the string.
EXAMPLE:
Input: 'Mr John Smith     ', 13
Output: 'Mr%20John%20Smith'
"""

def urlify(string, length):
    string = string[:length]
    return (string.replace(' ', '%20'))
print(urlify('Mr John Smith     ', 14))

"""
4. Palindrome Permutation:
Given a string, write a function to check if it is a permutation of a palindrome.
A palindrome is a word or a phrase that is the same forwards and backwards.
A permutation is a rearrangement of letters. The palindrome does not needs to be limited
to just dictionary words.
You can ignore casing and non-letter characters.
EXAMPLE:
Input: Tact Coa
Output: True (permutations: 'taco cat', 'atco cta', etc.)
"""


"""
5. One Away:
There are three types of edits that can be performed on strings: insert a character,
remove a character, or replace a character. Given two strings, write a function to
check if they are one(or zero edits) away.
EXAMPLE:
pale, ple -> true
pales, pale -> true
pale, bale -> true
pale, bake -> false
"""

def one_away(string1, string2):

    for s in string1:
        if string2:
            string2 = string2.strip(s)
        else:
            return True

    if len(string2) < 2:
        return True

    return False

print(one_away('pale', 'bake'))

"""
6. String Compression:
Implement a method to perform basic compression using the counts of repeated characters.
For example, the string aabcccccaaa would become a2b1c5a3. If the "compressed" string
would not become smaller than the original string, your method should return the original string.
You can assume the string has only uppercase and lowercase letters (a-z).
"""

def string_compression(string):

    output = string[0]
    count = 1
    for i in range(1,len(string)):
        if string[i] == string[i-1]:
            count+=1
        else:
            output += str(count)
            output += string[i]
            count = 1

    if len(output) >= len(string):
        return string
    else:
        return output+str(count)

print(string_compression('aaabccca'))

"""
7. Rotate Matrix:
Given an image represented by an N x N matrix, where each pixel in the image is represented
by an integer, write a method to rotate the image by 90 degrees.
Can you do this in place ?

Example 1:

Given input matrix = 
[
  [1,2,3],
  [4,5,6],
  [7,8,9]
],

rotate the input matrix in-place such that it becomes:
[
  [7,4,1],
  [8,5,2],
  [9,6,3]
]
Example 2:

Given input matrix =
[
  [ 5, 1, 9,11],
  [ 2, 4, 8,10],
  [13, 3, 6, 7],
  [15,14,12,16]
], 

rotate the input matrix in-place such that it becomes:
[
  [15,13, 2, 5],
  [14, 3, 4, 1],
  [12, 6, 8, 9],
  [16, 7,10,11]
]

"""
# This is with extra space
def rotate_image(matrix):
    output = []
    n = []
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            n.append(matrix[j][i])
        output.append(n[::-1])
        n = []

    return output
print(rotate_image([
  [ 5, 1, 9,11],
  [ 2, 4, 8,10],
  [13, 3, 6, 7],
  [15,14,12,16]
]))

# This is with In Place by modifying the given matrix
"""
Explanation


"""
def rotate_image_inplace(matrix):
    n = len(matrix)
    for i in range(n):
        for j in range(i, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]


    # i = 0, j = 0
    # matrix[0][0], matrix[0][0] = matrix[0][0], matrix[0][0] // 1
    # i = 0, j = 1
    # matrix[0][1] = 2, matrix[1][0] = 4
    # matrix[0][1],matrix[1][0] = matrix[1][0], matrix[0][1] // swap 2 and 4
    # i = 0. j = 3
    # matrix[0][2] = 3
    # matrix[0][2], matrix[2][0] = matrix[2][0], matrix[0][2] // swap 7 and 3

    # The idea was firstly transpose the matrix and then flip it symmetrically.
    # 1 2 3
    # 4 5 6
    # 7 8 9

    # after transpose, it will be swap(matrix[i][j], matrix[j][i]
    # 1 4 7
    # 2 5 8
    # 3 6 9

    #for i in range(n):
    #    for j in range((n//2)):
    #        matrix[i][j],matrix[i][n-1-j] = matrix[i][n-1-j],matrix[i][j]

    # Then flip the matrix horizontally. ( swap(matrix[i][j], matrix[i][len(matrix)- 1 - j])
    # 7 4 1
    # 8 5 2
    # 9 6 3

    for i in range(n):
        matrix[i] = matrix[i][::-1]

    # Or we can just reverse each row on the matrix

    return matrix

print(rotate_image_inplace([
  [1,2,3],
  [4,5,6],
  [7,8,9]
]))






















