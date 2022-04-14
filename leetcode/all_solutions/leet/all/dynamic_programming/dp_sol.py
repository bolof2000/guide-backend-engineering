from backports.functools_lru_cache import lru_cache


class Solutions:

    def permutations(self, l):

        """
        Backtracking solution
        we check if the len of the constructed permutation so far is equal to the len of the string
        we do a deep copy
        we then return to stop

        we loop through the given string
        check if the current index is used
        we skip
        we append the current letter to the path
        then set the used to True
        we dfs again on the input
        we pop from the path and mark it unsed by setting used to false

        """

        def dfs(path, used, res):
            if len(path) == len(l):
                res.append(path[:])  # do a deep copy
                return

            for i, letter in enumerate(l):
                if used[i]:
                    continue

                res.append(letter)
                used[i] = True
                dfs(path, used, res)
                path.pop()
                used[i] = False

        res = []
        dfs([], [False] * len(l), res)

        return res

    def letter_combinations_of_phone_number(self, digits):
        pass

    def fibo(self, n, memo):
        if n == 0 or n == 1:
            return n
        if n in memo:
            return memo[n]
        res = self.fibo(n - 1, memo) + self.fibo(n - 2, memo)

        memo[n] = res  # save it in memo before returning

        return res

    def wordBreak(self, s, worddict):

        def dfs(i):
            if i == len(s):  # i is the current position in the target we have matched so far
                return True  # establish when your calls should stop - always do this first

            for word in worddict:
                if s[i:].startswith(word):
                    if dfs(i + len(word)):
                        return True

            return False

        return dfs(0)

    def wordBreakMemo(self, s, wordDict):

        def dfs(i, memo):
            if i == len(s):
                return True

            if i in memo:
                return memo[i]

            ok = False
            for word in wordDict:
                if s[i:].startswith(word):
                    if dfs(i + len(word), memo):
                        return True
                        ok = True

            memo[i] = ok
            return ok

        return dfs(0, {})

    def decodeWays(self, digits):

        prefix = [str(i) for i in range(1, 27)]

        def dfs(i):
            if i == len(digits):
                return 1  # base case, you are done here

            ways = 0

            remaining = digits[i:]
            for pre in prefix:
                if remaining.startwith(pre):
                    ways += dfs(i + len(pre))
            return ways

        return dfs(0)

    def palindromePartitioning(self):
        pass

    def combinationSum(self, candidates, target):
        """
            Dedup
The way we dedup is to only use candidate numbers whose index in the array is >= last used number's index. In this example, when we are at the teal node, we don't want to look back and use any precedent candidate such as 2. This is because by DFS order we already explored subtracting 2 and during that traversal we have considered using 3 (blue nodes) .

        """

        def dfs(nums, start_index, remaining, path):
            if remaining == 0:
                res.append(path[:])
                return

            for i in range(start_index, len(nums)):
                num = nums[i]
                if remaining - num < 0:
                    continue

                dfs(nums, i, remaining - num, path + [num])

        res = []
        dfs(candidates, 0, target, [])
        return res

    def wordSearch(self):
        pass

    @staticmethod
    def canPartition(nums):
        n = len(nums)
        target = sum(nums) // 2
        if sum(nums) != 0 or max(nums) > target:
            return False

        used = [False] * n

        @lru_cache(None)
        def dfs(current_sum):
            if current_sum == target:
                return False  # we are done

            for i in range(n):
                if used[i]:
                    continue
                used[i] = True

                if dfs(current_sum + nums[i]):
                    return True
                    used[i] = False

            return False

        return dfs(0)
