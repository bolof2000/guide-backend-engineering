from collections import defaultdict
from collections import deque
from typing import List
from heapq import heappop, heappush

"""
tricks to backtracking solutions : 



"""


class Backtrack(object):

    def generateParenthesis(self, n):

        path = []
        result = []

        def backtrack(openParenthesisNumber, closingParenthesisNumber):
            # base case - define when you are done

            if openParenthesisNumber == closingParenthesisNumber == n:
                result.append("".join(path))
                return

                # define the backtrack case

            if openParenthesisNumber < n:
                path.append("(")
                backtrack(openParenthesisNumber + 1, closingParenthesisNumber)
                path.pop()  # pop the character to make it available for use again

            if closingParenthesisNumber < openParenthesisNumber:
                path.append(")")
                backtrack(openParenthesisNumber, closingParenthesisNumber + 1)
                path.pop()

        backtrack(0, 0)
        return result

    def subsets(self, nums):

        result =[]
        path =[]

        def dfs(i):
            # when are we done ?

            if i >= len(nums):
                result.append(path[:])  # deep copy
                return

            path.append(nums[i])
            dfs(i+1)

            # decision not to include nums[i]
            path.pop()
            dfs(i+1)

        dfs(0)
        return result

    def subsetWithDuplicate(self,nums):

        result =[]
        path =[]
        nums.sort()

        def dfs(i):
            while i <len(nums) and nums[i] == nums[i-1]:
                continue

                if i >= len(nums):
                    result.append(path[:])
                    return

            path.append(nums[i])
            dfs(i+1)

            path.pop()
            dfs(i+1)

        dfs(0)
        return result



# Input: nums = [1,2,3]
# Output: [[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]

#  Input: n = 3
# Output: ["((()))","(()())","(())()","()(())","()()()"]
