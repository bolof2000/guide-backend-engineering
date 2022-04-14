"""
Pattern:

Identify the state
Draw the state-space tree
DFS/backtrack on the state-space tree


What state do we need to know whether we have reached a solution (and using it to construct a solution if the problem asks for it).
 In the above permutation example, we need to keep track of the letters we have already selected when we do DFS.

What state do we need to decide which child nodes should be visited next and which ones should be pruned.
In the above permutation example, we have to know what are the letters left that we can still use (since each letter can only be used once).


"""
from typing import List


def permute(string: str):
    def dfs01(used, path, res):
        # state
        if len(path) == len(string):
            res.append(path)
            return
        for i, char in enumerate(string):
            if used[i]:
                continue
            path.append(char)
            dfs01(used[i], path, res)
            path.pop()
            used[i] = False

    res = []
    dfs01([False] * len(string), [], res)
    return res


def letterCombinations(string):
    # base case
    keyword_map = {
        "2": "abc",
        "3": "def",
        "4": "ghi",
        "5": "jkl",
        "6": "mno",
        "7": "pqrs",
        "8": "yuv",
        "9": "wxyz"
    }

    def dfs02(path, index, res):
        if len(path) == len(string):
            res.append(path)
            return
        key = string[index]

        for char in keyword_map[key]:
            path.append(char)
            dfs02(path, index + 1, res)
            path.pop()

        res = []
        dfs02([], 0, res)

        return res


def wordBreak(s: str, words: List[str]) -> bool:
    def dfs03(index):
        if index == len(str):
            return True

        for word in words:
            if s[index:].startswith(word):
                if dfs03(index + len(word)):
                    return True

        return False

    return dfs03(0)


def combination_sums(candidates: List[int], target: int) -> List[List[int]]:
    def dfs(index, nums: List[int], path: List[int], temp, res):
        if temp == 0:
            res.append(path[:])
            return

        for i in range(index, len(nums)):
            num = nums[i]
            if temp - num < 0:
                continue

            dfs(i, nums, path + [num], temp - num, res)

    res = []
    dfs(0, candidates, [], target, res)
    return res


def subsets(nums: List[int]):
    n = len(nums)
    res = []

    def dfs(index, path):
        if index == n:
            res.append(path)
            return
        dfs(index + 1, path + [nums[index]])
        dfs(index + 1, path)

    dfs(0, [])
    return res


if __name__ == '__main__':
    num01 = [2, 3, 6, 7]
    print(combination_sums(num01, 7))
