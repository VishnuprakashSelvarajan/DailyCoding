"""
This problem was asked by Amazon.

Given an array of numbers, find the maximum sum of any contiguous subarray of the array.

For example, given the array [34, -50, 42, 14, -5, 86],
the maximum sum would be 137, since we would take elements 42, 14, -5, and 86.

Given the array [-5, -1, -8, -9], the maximum sum would be 0, since we would not take any elements.

Do this in O(N) time.
"""
def max_sum_sub_array(nums):
    maxcurr = nums[0]
    maxglobal = nums[0]
    for i in range(1, len(nums)):
        maxcurr = max(nums[i], maxcurr + nums[i])
        maxglobal = max(maxcurr, maxglobal)
    return maxglobal

print(max_sum_sub_array([34, -50, 42, 14, -5, 86]))





