''' Leetcode problem 219 '''
"""
Given an array of integers and an integer k, find out whether there are two distinct 
indices i and j in the array such that nums[i] = nums[j] and the absolute difference between i and j is at most k.
Example 1:
Input: nums = [1,2,3,1], k = 3
Output: true
Example 2:
Input: nums = [1,0,1,1], k = 1
Output: true
Example 3:
Input: nums = [1,2,3,1,2,3], k = 2
Output: false
"""


def containsNearbyDuplicate(nums, k):
    # Brute Force -- Not efficient
    '''
    if len(nums) == 0:
        return False

    if k == 0:
        return False

    if len(nums) == 2:
        if k == 2 and nums[0] == nums[1]:
            return True

    for i in range(len(nums)):
        for j in range(i+1, len(nums)):
            if nums[i] == nums[j]:
                if (j - i) <= k:
                    return True

    return False
    '''

    # Efficient one
    duplicates = {}
    for idx, val in enumerate(nums):
        if val in duplicates and idx - duplicates[val] <= k:
            return True
        else:
            duplicates[val] = idx

    return False

print(containsNearbyDuplicate([1,2,3,1,2,3], 2))