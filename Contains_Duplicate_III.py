'''Leetcode 220. Contains Duplicate III'''

"""

Given an array of integers, find out whether there are two distinct indices i and j in the array such that the absolute difference between nums[i] and nums[j] is at most t and the absolute difference between i and j is at most k.

Example 1:

Input: nums = [1,2,3,1], k = 3, t = 0
Output: true
Example 2:

Input: nums = [1,0,1,1], k = 1, t = 2
Output: true
Example 3:

Input: nums = [1,5,9,1,5,9], k = 2, t = 3
Output: false
"""


class Solution:
    def containsNearbyAlmostDuplicate(self, nums: List[int], k: int, t: int) -> bool:

        size = len(nums)

        if t == 0 and len(nums) == len(set(nums)):
            # Quick response for special case on t = 0
            # t = 0 requires at last one pair of duplicate elements
            return False

        for idx, cur_val in enumerate(nums):

            for j in range(idx + 1, idx + k + 1):

                if j >= size:
                    # avoid index out of boundary
                    break

                if abs(cur_val - nums[j]) <= t:
                    # hit:
                    # i != j, | i-j | <= k
                    # | nums[i] - nums[j] | <= t
                    return True

        return False

