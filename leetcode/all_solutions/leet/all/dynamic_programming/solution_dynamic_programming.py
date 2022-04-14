class Solutions:

    def wordBreak(self, s, wordDict):

        """

        :param s:
        :param wordDict:
        :return:
        """
        memo = {}

        def dfs(i):

            if i == len(s):
                return True  # we are done

            if i in memo:
                return memo[i]

            ok = False

            for word in wordDict:
                if s[i:].startswith(word):
                    if dfs(i + len(word)):
                        return True
                        ok = True
                        break

            memo[i] = ok

            return ok

        return dfs(0)

    def fibo(self, n, memo):

        if n in memo:
            return memo[n]

        if n == 0 or n == 1:
            return n

        res = self.fibo(n - 1, memo) + self.fibo(n - 2, memo)

        memo[n] = res

        return res

    def numDecodingsWithoutMemo(self, s):

        """
        Solve without memoization to reduce time complexity
        Approach is to:
        1:  Identify the states
        2: build the space-state tree
         Keep track of visited digits in index i
         when i = len of digits, task is finished
         to reduce time complexity use memo dic to store visited digits and then check in the dictionary
        :param s:
        :return:
        """
        prefix = [str(i) for i in range(1, 27)]  # represents all alha that we can matched

        def dfs(i):
            if i == len(s):
                return 1  # we are done

            ways = 0
            remaining = s[i:]

            for pre in prefix:
                if remaining.startswith(pre):
                    ways += dfs(i + len(pre))

            return ways  # we could have optimized by storing ways in memo before returning it

        return dfs(0)

    def numDecodingsWithMemo(self, s):

        prefix = [str(i) for i in range(1, 27)]

        def dfs(i, memo):
            if i == len(s):
                return 1

            if i in memo:
                return memo[i]

            ways = 0
            remaining = s[i:]

            for pre in prefix:
                if remaining.startswith(pre):
                    ways += dfs(i + len(pre), memo)

            memo[i] = ways

            return ways

        return dfs(i, {})


    def wordBreakWithoutMemo(self, s, wordDict):

        """
        Same approach to decode
        we return true or false and to optimize we store true in the memo
        :param s:
        :param wordDict:
        :return:
        """

        def dfs(i):
            if i == len(s):
                return True

            for word in wordDict:
                if s[i:].startswith(word):
                    if dfs(i+len(word)):
                        return True

            return False
        return dfs(0)

    def wordBreakWithMemo(self, s, wordDict):

        def dfs(i,memo):
            if i == len(s):
                return True

            if i in memo:
                return memo[i]
            
            ok = False

            for word in wordDict:
                if s[i:].startswith(word):
                    if dfs(i+len(word),memo):
                        return True
                        ok = True
                        break

            memo[i] = ok
            return ok

        return dfs(0,{})




test = Solutions()
s = "leetcode"
words = ["leet","codee"]
print(test.wordBreakWithoutMemo(s,words))