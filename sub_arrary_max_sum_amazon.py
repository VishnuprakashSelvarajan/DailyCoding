"""
This problem was asked by Amazon.

Given an array of numbers, find the maximum sum of any contiguous subarray of the array.

For example, given the array [34, -50, 42, 14, -5, 86],
the maximum sum would be 137, since we would take elements 42, 14, -5, and 86.

Given the array [-5, -1, -8, -9], the maximum sum would be 0, since we would not take any elements.

Do this in O(N) time.
"""
def max_sum_sub_array(array):

    max_sum = sum(array)
    if max_sum <= 0:
        return 0

    for i in range(len(array)):
        temp_sum = array[i]
        for j in range(i+1, len(array)):
            temp_sum += array[j]

        max_sum = max(max_sum, temp_sum)

    return max_sum

print(max_sum_sub_array([34, -50, 42, 14, -5, 86]))




