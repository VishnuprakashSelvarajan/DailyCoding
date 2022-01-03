"""
Best Time to Buy and Sell Stock
"""

"""
Say you have an array for which the ith element is the price of a given stock on day i.

If you were only permitted to complete at most one transaction
(i.e., buy one and sell one share of the stock), design an algorithm to find the maximum profit.

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

"""
def maxProfit(prices):

    maxprofit = 0
    buy_day = 0
    if len(prices) < 2:
        return maxprofit
    while buy_day < len(prices) - 1:
        next_day = buy_day + 1
        temp_maxprofit = max(prices[next_day:])
        profit = temp_maxprofit - prices[buy_day]
        maxprofit = max(maxprofit, profit)
        buy_day += 1

    return maxprofit

print(maxProfit([2,1,2,1,0,1,2]))

def max_profit(prices):
    minPrice = math.inf
    maxProfit = 0
    for i in range(len(prices)):
        if prices[i] <= minPrice:
            minPrice = prices[i]
        if prices[i] - minPrice > 0:
            maxProfit = max((prices[i] - minPrice), maxProfit)

    return maxProfit

print(max_profit([2,1,2,1,0,1,2]))
