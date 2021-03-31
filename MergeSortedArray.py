'''
Given two sorted integer arrays nums1 and nums2, merge nums2 into nums1 as one sorted array.

The number of elements initialized in nums1 and nums2 are m and n respectively. You may assume that nums1 has a size equal to m + n such that it has enough space to hold additional elements from nums2.



Example 1:

Input: nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3
Output: [1,2,2,3,5,6]


Example 2:

Input: nums1 = [1], m = 1, nums2 = [], n = 0
Output: [1]

'''


def merge(nums1, m, nums2, n):
    """
    Do not return anything, modify nums1 in-place instead.
    """
    print(nums1 + nums2)
    for i in range(n):
        nums1[m] = nums2[i]
        m += 1
    print(nums1)
    for i in range(len(nums1)):
        for j in range(len(nums1)):
            if nums1[i] < nums1[j]:
                nums1[j], nums1[i] = nums1[i], nums1[j]

    return nums1

print(merge(nums1= [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3))

