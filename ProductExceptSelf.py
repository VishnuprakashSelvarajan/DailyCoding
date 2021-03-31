'''
Product of Array Except Self
Given an integer array nums, return an array answer such that answer[i] is equal to the product of all the elements of nums except nums[i].

The product of any prefix or suffix of nums is guaranteed to fit in a 32-bit integer.

Example 1:

Input: nums = [1,2,3,4]
Output: [24,12,8,6]

Example 2:

Input: nums = [-1,1,0,-3,3]
Output: [0,0,9,0,0]

'''


def productExceptSelf(nums):
    output = []
    for i in range(len(nums)):
        out = nums.pop(0)
        result = 1
        for x in nums:
            result = result * x
        output.append(result)
        nums.append(out)

    return output
