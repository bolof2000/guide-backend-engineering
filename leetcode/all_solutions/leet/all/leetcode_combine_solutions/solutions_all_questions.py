from collections import defaultdict

from leetcode.listnode import ListNode


class Solutions:

    def threeSum(self, nums):
        result = []
        nums.sort()
        for i in range(len(nums) - 2):
            if i == 0 or (i > 0 and nums[i] != nums[i - 1]):

                left = i + 1
                right = len(nums) - 1

                while left < right:

                    temp = nums[left] + nums[right] + nums[i]
                    if temp == 0:
                        result.append([nums[i], nums[left], nums[right]])

                        while left < right and nums[left] == nums[left + 1]:
                            left += 1

                        while left < right and nums[right] == nums[right - 1]:
                            left -= 1

                        left += 1
                        right -= 1

                    elif temp > 0:
                        right -= 1
                    else:
                        left += 1

        return result

    def exist(self, board, word):
        for row in range(len(board)):
            for col in range(len(board[0])):
                if board[row][col] == word[0] and self.dfs(board, row, col, word):
                    return True

        return False

    def dfs(self, board, row, col, word):
        if len(word) == 0:
            return True
        if row < 0 or row >= len(board) or col < 0 or col > len(board[0]) or board[row][col] != word[0]:
            return False

        temp = board[row][col]
        board[row][col] = ' '
        if (self.dfs(board, row - 1, col, word[1:])
                or self.dfs(board, row + 1, col, word[1:])
                or self.dfs(board, row, col - 1, word[1:])
                or self.dfs(board, row, col + 1, word[1:])
        ):
            return True

        board[row][col] = temp

        return False

    def productArrayExceptSelf(self, nums):

        result = [1] * len(nums)
        multiply = 1
        for i in range(0, len(nums)):
            result[i] = result[i] * multiply
            multiply = multiply * nums[i]

        multiply = 1

        for i in range(len(nums) - 1, -1, -1):
            result[i] = result[i] * multiply
            multiply = multiply * nums[i]

        return result

    def subArrayEqualSum(self, nums, k):
        mapp = defaultdict(int)
        mapp.setdefault(0, 1)
        summ = 0
        count = 0
        for num in nums:
            summ += num
            temp = summ - k

            if temp in mapp:
                count += mapp.get(temp)

            mapp[summ] += 1

        return count

    def decodeString(self, s):
        stack = []
        for char in s:
            if char != ']':
                stack.append(char)
            else:

                res = []
                while len(stack) > 0 and str.isalpha(stack[-1]):
                    res.insert(0, stack.pop())

                final_string = "".join(res)
                stack.pop()

                digit = []
                while len(stack) > 0 and str.isdigit(stack[-1]):
                    digit.insert(0, stack.pop())

                final_digit = "".join(digit)
                int_digit = int(final_digit)

                while int_digit > 0:
                    for charr in final_string:
                         stack.append(charr)
                    int_digit -= 1

        output = []
        while len(stack) > 0:
            output.insert(0, stack.pop())

        return "".join(output)

    def houseRobber(self,nums):
        if len(nums) == 0:
            return 0
        if len(nums) ==1:
            return nums[0]
        if len(nums) ==2:
            return max(nums[0],nums[1])

        dp = [nums[0],max(nums[0],nums[1])]

        for i in range(2,len(nums)):
            dp.append(max(dp[i-2]+nums[i],dp[i-1]))

        return dp.pop()

    def removeElements(self,head:ListNode,val:int)-> ListNode:
        pass


test = Solutions()
print(test.productArrayExceptSelf([1, 2, 3, 4]))
