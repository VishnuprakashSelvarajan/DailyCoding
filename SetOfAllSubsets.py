"""
This problem was asked by Google.

The power set of a set is the set of all its subsets. Write a function that, given a set, generates its power set.

For example, given the set {1, 2, 3}, it should return {{}, {1}, {2}, {3}, {1, 2}, {1, 3}, {2, 3}, {1, 2, 3}}.

You may also use a list or array to represent a set.
"""


def all_subsets(given_array):
    subset = [0 for i in range(len(given_array))]
    return helper(given_array, subset, 0)


def helper(given_array, subset, i):
    output = []
    if i == len(given_array):
        print(subset)
    else:
        subset[i] = 'null'
        helper(given_array, subset, i + 1)
        subset[i] = given_array[i]
        helper(given_array, subset, i + 1)

    return output


print(all_subsets([1, 2, 3]))

"""
Another method similar
"""


def subsets(nums: List[int]) -> List[List[int]]:
    output = []
    helper(nums, output, [], 0)
    return output


def helper(nums, output, curr, index):
    output.append(list(curr))
    for i in range(index, len(nums)):
        curr.append(nums[i])
        helper(nums, output, curr, i + 1)
        curr.pop()


print(subsets[1, 2, 3])
