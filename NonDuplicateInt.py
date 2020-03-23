"""
This problem was asked by Google.

Given an array of integers where every integer occurs three times except for one integer, which only occurs once,
find and return the non-duplicated integer.

For example, given [6, 1, 3, 3, 3, 6, 6], return 1. Given [13, 19, 13, 13], return 19.

Do this in O(N) time and O(1) space.

"""


def non_duplicate(arr):
    set_arr = {}

    for num in arr:
        if num in set_arr:
            set_arr[num] += 1
        else:
            set_arr[num] = 1
    print(set_arr)
    return [key for key, val in set_arr.items() if val == 1]


print(non_duplicate([6, 1, 3, 3, 3, 6, 6]))
