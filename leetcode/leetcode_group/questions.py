import collections
from collections import defaultdict
from typing import List
from math import inf
from collections import deque
from heapq import heappush, heappop


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val


class TreeNode:
    def __init__(self, left=None, right=None, val=0):
        self.left = left
        self.right = right


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
                            right -= 1

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
                if board[row][col] == word[0] and self.dfs(board, row, col, word,count):
                    return True

        return False

    def dfs(self, board, row, col, word,count):
        if len(word) == count:
            return True
        if row < 0 or row >= len(board) or col < 0 or col > len(board[0]) or board[row][col] != word[count]:
            return False
        temp = board[row][col]
        board[row][col] = ' '
        found = self.dfs(board, row - 1, col, word,count+1)or self.dfs(board, row + 1, col, word,count+1)   or self.dfs(board, row, col - 1, word,count+1)  or self.dfs(board, row, col + 1, word,count+1)
        board[row][col] = temp
        return found

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

    def houseRobber(self, nums):
        if len(nums) == 0:
            return 0
        if len(nums) == 1:
            return nums[0]
        if len(nums) == 2:
            return max(nums[0], nums[1])

        dp = [nums[0], max(nums[0], nums[1])]

        for i in range(2, len(nums)):
            dp.append(max(dp[i - 2] + nums[i], dp[i - 1]))

        return dp.pop()
    def house_robber_two(self,nums):
        return max(self.houseRobber(nums[1:]),self.houseRobber(nums[:-1]))

    def removeElements(self, head: ListNode, val: int) -> ListNode:

        dummy = ListNode(0)
        output = dummy
        dummy.next = head
        while head is not None:
            if head.val == val:
                head = head.next
                dummy.next = head
            else:
                head = head.next
                dummy = dummy.next

        return output.next

    @staticmethod
    def longestPalindrome(s: str) -> int:
        stack = []
        count = 0
        for char in s:
            if char not in stack:
                stack.append(char)
            else:
                if len(stack) > 0 and char in stack:
                    stack.pop()
                    count += 1

        result = count * 2
        if len(stack) > 0:
            result += 1

        return result

    @staticmethod
    def longestOnes(nums: List[int], k: int):

        """
         A = [1,1,1,0,0,0,1,1,1,1,0]  k = 2
        """

        window_start = 0
        right = 0
        while right < len(nums):
            if nums[right] == 0:
                k -= 1

            if k < 0:
                if nums[right] == 0:
                    k += 1
                    window_start += 1

            right += 1

        return right - window_start

    def pivotIndex(self, nums: List[int]) -> int:

        summation = sum(nums)

        accum = 0
        for i in range(len(nums)):
            if summation - nums[i] - accum == accum:
                return i
            accum += nums

        return accum

    def fib(self, n):
        if n < 2:
            return n
        return self.fib(n - 1) + self.fib(n - 2)

    def fib(self, n, memo):

        if n < 2:
            return memo[n]

        res = self.fib(n - 1, memo) + self.fib(n - 2, memo)

        memo[n] = res

        return res

    def rotateArray(self, nums, k):
        k = k % len(nums)

        self.helper(nums, 0, len(nums) - 1)
        self.helper(nums, 0, k - 1)
        self.helper(nums, k, len(nums) - 1)

        return nums

    def helper(self, nums, left, right):

        while left < right:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
            right -= 1

    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:

        dummy = ListNode(0)
        dummy.next = head
        slow = dummy
        fast = dummy

        while n > 0:
            fast = fast.next
            n -= 1

        while fast and fast.next is not None:
            slow = slow.next
            fast = fast.next

        slow.next = slow.next.next

        return dummy.next

    def findMaxConsecutiveOnes(self, nums):
        window_start = 0
        max_len = 0
        for i in range(len(nums)):

            if nums[i] == 1:
                max_len = max(max_len, i - window_start + 1)
            else:
                window_start = i + 1

        return max_len

    def removeDuplicates(self, s):
        stack = []
        for char in s:
            if char not in stack:
                stack.append(char)
            else:
                if len(stack) > 0 and char == stack[-1]:
                    stack.pop()
        return "".join(stack)

    def searchMatrix(self, matrix, target):

        if len(matrix) == 0:
            return False

        row = len(matrix)
        col = len(matrix[0])
        left = 0
        right = row * col - 1

        while left <= right:
            mid = (left + right) // 2
            midNum = matrix[mid / col][mid % col]
            if target == midNum:
                return True

            elif target > midNum:
                left = mid + 1
            else:
                right = mid - 1

        return False

    def searchMatrixII(self, matrix, target):

        if len(matrix) == 0:
            return False

        row = len(matrix)
        col = len(matrix[0])

        currentRow = 0
        currentCol = col - 1
        while currentRow < row and currentCol >= 0:
            if matrix[currentRow][currentCol] == target:
                return True
            if matrix[currentRow][currentCol] > target:
                currentCol -= 1
            else:
                currentRow += 1
        return False

    def minSubArrayLen(self, s, nums):
        if len(nums) == 0:
            return 0
        window_start = 0
        minLen = float('inf')
        summation = 0
        for i in range(len(nums)):
            summation += nums[i]
            while summation >= s:
                min(minLen, i - window_start + 1)

                summation -= nums[window_start]

                window_start += 1

        if minLen == float('inf'):
            return 0

        return minLen

    def transposeMatrix(self, matrix):

        row = len(matrix)
        col = len(matrix[0])

        result = [col][row]
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                result[j][i] = matrix[i][j]

        return result

    def maxProfit(self, prices):
        profit = 0
        cheapest_price = prices[0]
        for price in prices:
            cheapest_price = min(price, cheapest_price)
            current_profit = price - cheapest_price
            profit = max(current_profit, profit)

        return profit

    def isSymmetric(self, root: TreeNode) -> bool:

        # what is the return value
        #  what is the state to pass to the child
        # the state here is to compare left and right node

        # the base case needs to do this and then recursive calls do that for the child

        def dfs(left: TreeNode, right: TreeNode) -> bool:

            if left is None and right is None:
                return True
            if left is None or right is None or left.val != right.val:
                return False

            return dfs(left.left, right.right) and dfs(left.right, right.left)

        if root is None:
            return True

        return dfs(root.left, root.right)

    def invertBinaryTree(root: TreeNode):

        if not root:
            return None

        def dfs(root: TreeNode):
            temp = root.left
            root.left = root.right
            root.right = temp

            dfs(root.left)
            dfs(root.right)

        dfs(root)
        return root

    def mergeTrees(self, t1: TreeNode, t2: TreeNode) -> TreeNode:

        def dfs(t1, t2):
            if t1 is None and t2 is None:
                return None

            if t1 is None:
                return t2
            if t2 is None:
                return t1

            t3 = TreeNode(t1.val + t2.val)

            dfs(t1.left, t2.left)
            dfs(t1.right, t2.right)

            return t3

        return dfs(t1, t2)

    def palidroneLinkedList(self, head: ListNode) -> bool:

        res = []
        current = head
        while head is not None:
            res.append(head.val)
            head = head.next

        left = 0
        right = len(res) - 1
        while left < right:
            if res[left] != res[right]:
                return False

            left += 1
            right -= 1

        return True

    def inOrderTraversal(self, root: TreeNode):

        def dfs(root, result):
            if not root:
                return []

            dfs(root.left, result)
            result.append(root.val)
            dfs(root.right, result)

        res = []
        dfs(root, res)

        return res

    def minimumAddToMakeValid(self, s):

        if len(s) == 0:
            return 0

        stack = []

        for char in s:
            if char == '(':
                stack.append(char)
            elif len(stack) > 0 and stack[-1] == '(':
                stack.pop()
            else:
                stack.append(char)

        return len(stack)

    def maxDepth(self, root: TreeNode) -> int:

        def dfs(root):
            if not root:
                return 0

            return max(dfs(root.left), dfs(root.right)) + 1

        return dfs(root)

    def reverse_string(self, s):
        res = []
        for i in range(len(s) - 1, -1, -1):
            res.append(s[i])

        return res

    def middleNode(self, root: ListNode) -> ListNode:
        slow = root
        fast = root

        while fast is not None and fast.next is not None:
            slow = slow.next
            fast = fast.next.next

        return slow

    def mergeTwoSortedList(self, l1: ListNode, l2: ListNode) -> ListNode:
        dummy = ListNode(0)
        output = dummy

        while l1 is not None and l2 is not None:
            if l1.val > l2.val:
                dummy.next = l2
                l2 = l2.next

            else:
                dummy.next = l1
                l1 = l1.next

            dummy = dummy.next

        if l1 is not None:
            dummy.next = l1

        if l2 is not None:
            dummy.next = l2

        return output.next

    def threeSumClosest(self, nums, target):
        closest = nums[0] + nums[1] + nums[len(nums) - 1]
        nums.sort()
        for i in range(len(nums) - 2):
            left = i + 1
            right = len(nums) - 1

            while left < right:

                current = nums[left] + nums[i] + nums[right]

                if abs(target - current) < abs(target - closest):
                    closest = current
                elif target == current:
                    return target
                elif target > current:
                    left += 1
                else:
                    right -= 1

        return closest

    def searchBST(self, root: TreeNode, val: int) -> TreeNode:

        if root.val == val:
            return root

        elif root.val > val:
            return self.searchBST(root.left, val)
        else:
            return self.searchBST(root.right, val)

    def reverse_only_letters(self, s):
        """
        input :  ab-cd
        """
        stack = []
        res = []
        for char in s:
            if str.isalpha(char):
                stack.append(char)
        print(stack)

        for ca in s:
            if str.isalpha(ca):
                res.append(stack.pop())

            else:
                res.append(ca)
        print(res)

        return "".join(res)

    def rangeSumBST(self, root: TreeNode, l: int, r: int) -> int:

        def dfs(root):
            summation = 0
            if not root:
                return 0
            if l <= root.val <= r:
                summation += root.val

            summation += dfs(root.left)
            summation += dfs(root.right)

            return summation

        return dfs(root)

    def sortArrayByParity(self, nums):

        even = []
        odd = []
        for num in nums:
            if num % 2 == 0:
                even.append(num)
            if num % 2 == 1:
                odd.append(num)
        return even + odd

    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        dummy = ListNode(0)
        result = dummy

        carry = 0
        while l1 is not None or l2 is not None or carry != 0:
            if l1 is not None:
                carry += l1.val
                l1 = l1.next

            if l2 is not None:
                carry += l2.val
                l2 = l2.next

            dummy.next = ListNode(carry % 10)
            carry = carry / 10

            dummy = dummy.next

        return result.next

    def encode(self, str):
        """
        str = ["Kevin","is","great"]
        output : "5/kevin2/is5/great"
        """
        encoded = ""
        for word in str:
            length = len(word)
            encoded += length + "/" + word

        return encoded

    def decode(self, strs: str) -> List:

        position = 0
        decoded = []
        while position < len(strs):
            slash_position = strs.index("/", position)
            length = int(strs[slash_position - 1])
            position = slash_position + 1

            decoded.append(strs[position:position + length])
            position += length

        return decoded

    def climbingStairs(self, n):

        if n <= 3:
            return n

        ways = [0, 1, 2, 3]
        for i in range(4, n + 1):
            ways.append(ways[i - 1] + ways[i - 2])

        return ways.pop()

    def houseRobber(self, nums):

        if len(nums) == 0:
            return 0

        if len(nums) == 1:
            return nums[0]
        if len(nums) == 2:
            return max(nums[1], nums[2])
        dp = [nums[0], max(nums[1], nums[2])]
        for i in range(2, len(nums)):
            dp.append(max(nums[i] + dp[i - 2], dp[i - 1]))

        return dp.pop()  # or dp[-1]

    def maxSubArray(self, nums):

        maxSum = nums[0]
        currentSum = nums[0]
        for i in range(1, len(nums)):
            currentSum = max(nums[i] + currentSum, nums[i])
            maxSum = max(currentSum, maxSum)

        return maxSum

    def canAttentMeetings(self, intervals):
        intervals.sort()

        for i in range(len(intervals) - 1):
            if intervals[i + 1][0] < intervals[i][1]:
                return False

    def merge(self, intervals):
        if not intervals:
            return []
        intervals.sort(key=lambda interval: interval[0])
        res = [intervals[0]]
        for current in intervals:
            last_interval = [-1]
            if current[0] <= last_interval[1]:
                last_interval[1] = max(current[1], last_interval[1])
            else:
                res.append(current)
        return res

        return True

    def spiralMatrix(self, matrix):
        result = []
        if not matrix:
            return []

        top = 0
        bottom = len(matrix) - 1

        left = 0
        right = len(matrix[0]) - 1
        direction = "right"

        while top <= bottom and left <= right:

            if direction == "right":
                for i in range(left, right + 1):
                    result.append(matrix[top][i])

                top += 1
                direction = "down"
            elif direction == "down":
                for i in range(top, bottom - 1):
                    result.append(matrix[i][right])

                right -= 1

                direction = "right"

            elif direction == "right":
                for i in range(bottom, left - 1, -1):
                    result.append(matrix[bottom][i])
                bottom -= 1

                direction = "up"

            elif direction == "up":
                for i in range(left, top - 1):
                    result.append(matrix[i][left])
                left -= 1
                direction = "right"
        return result

    """
    Graph Questions here . Delete and write again till you fully understands it 
    """

    def countConnectedNodes(self, N, edges):

        adjacent_list = {}
        visited = {}
        count = 0

        for vertex in range(N):
            adjacent_list[vertex] = []
            visited[vertex] = False

        for edge in edges:
            v1 = edge[0]
            v2 = edge[1]

            adjacent_list[v1].append(v2)
            adjacent_list[v2].append(v1)

        def dfs(vertex):
            visited[vertex] = True

            for neighbor in adjacent_list[vertex]:
                if not visited[neighbor]:
                    dfs(neighbor)

        for vertex in range(N):
            if not visited[vertex]:
                dfs(vertex)
                count += 1

        return count

    def canFinish(self, numCourses, prerequisites):

        adj_list = {}
        visited = {}

        for vertex in range(numCourses):
            adj_list[vertex] = []
            visited[vertex] = "white"

        for edge in prerequisites:
            v1 = edge[0]
            v2 = edge[1]
            adj_list[v1].append(v2)

        def dfs(vertex):
            visited[vertex] = "gray"
            for neighbor in adj_list[vertex]:
                if visited[neighbor] == "gray" and dfs(neighbor):
                    return True

            if visited[neighbor] == "white" and dfs(neighbor):
                return True

            visited[vertex] = "black"
            return False

        for vertex in range(numCourses):
            if visited[vertex] == "white":
                dfs(vertex)
                return False

        return True

    def maxDepthSolution(self, root: TreeNode) -> int:
        def dfs(root):
            if not root:
                return 0
            return max(dfs(root.left), dfs(root.right)) + 1

        return dfs(root)

    def visible_tree_node(self, root: TreeNode) -> int:

        def dfs(root, max_so_far):

            if not root:
                return 0

            total = 0
            if root.val > max_so_far:
                total += 1

            total += dfs(root.left, max(max_so_far, root.val))
            total += dfs(root.right, max(max_so_far, root.val))

            return total

        return dfs(root, -float('inf'))

    def validateBST(self, root: TreeNode) -> bool:

        def dfs(root: TreeNode, min_val: int, max_val: int) -> bool:

            if not root:
                return True
            if root.val <= min_val and root.val >= max_val:
                return False

            return dfs(root.left, min_val, root.val) and dfs(root.right, root.val, max_val)

        return dfs(root, -inf, inf)

    def serialize(self, root: TreeNode):

        res = []

        def dfs(root):
            if not root:
                res.append('x')
                return
            res.append(root.val)

            dfs(root.left)
            dfs(root.right)

        dfs(root)

        return ' '.join(res)

    def deserialize(self, s):

        def dfs(nodes):
            val = next(nodes)
            if val == 'x':
                return

            current = TreeNode(int(val))

            current.left = dfs(nodes)
            current.right = dfs(nodes)

            return current

        return dfs(iter(s.split()))

    def level_order_traversal(self, root):
        res = []
        queue = deque([root])
        while len(queue) > 0:

            n = len(queue)
            new_level = []
            for _ in range(n):
                node = queue.popleft()
                new_level.append(node.val)

            for child in [root.left, root.right]:
                if child is not None:
                    queue.append(child)

        res.append(new_level)

        return res

    def zig_zag(self, root):

        res = []
        if not root:
            return res
        left_to_right = True
        queue = deque([root])

        while len(queue) > 0:
            n = len(queue)
            new_level = []
            for _ in range(n):
                node = queue.popleft()
                new_level.append(node.val)
            for child in [node.left, node.right]:
                if child is not None:
                    queue.append(child)

            if not left_to_right:
                new_level.reverse()
            res.append(new_level)

            left_to_right = not left_to_right

        return res

    def permute(self, s):

        def dfs(path, used, res):

            if len(path) == len(s):
                res.append(path[:])
                return
            for i, letter in enumerate(s):
                if used[i]:
                    continue
                path.append(letter)
                used[i] = True

                dfs(path, used, res)
                path.pop()
                used[i] = False

        res = []
        dfs([], [False] * len(s), res)

        return res

    def wordBreak(self, s, wordDict):

        def dfs(i, memo):
            # base case

            if i == len(s):
                return True

            if i in memo:
                return memo[i]

            ok = False

            for word in wordDict:
                if s[i:].startswith(word):
                    if dfs(i + len(word), memo):
                        ok = True

            memo[i] = ok
            return ok

        return dfs(0, {})

    def wordBreakNoMemo(self, s, dictWord):

        def dfs(i):
            if i == len(s):
                return True

            for word in dictWord:
                if s[i:].startswith(word):
                    if dfs(i + len(word)):
                        return True

            return False

        dfs(0)

    def decode_ways(self, digits):
        prefixes = [str(i) for i in range(1, 27)]

        def dfs(i, memo):

            if i in memo:
                return memo[i]
            ways = 0
            for prefix in prefixes:
                if digits[i:].startswith(prefix):
                    ways += dfs(i + len(prefix), memo)

            memo[i] = ways

            return ways

        return dfs(0, {})

    def combination_sum(self, nums, target):
        pass

    def validate_ip4(self, ipaddress):
        pass

    def validate_ip4(self, ipaddres):
        pass


def minCost(cost, s):
    result = 0
    pre = 0
    for i in range(1, len(s)):
        if s[pre] != s[i]:
            pre = i

        else:

            result += max(cost[pre], cost[i])
            if cost[pre] < cost[i]:
                pre = i
    return result


def numIslands(grid):
    if (grid == None or len(grid) == 0):
        return 0
    count = 0
    for row in range(len(grid)):
        for col in range(len(grid[0])):
            if grid[row][col] == "1":
                count += 1
                dfs(grid, row, col)

    return count


def dfs(grid, row, col):
    if row < 0 or row >= len(grid) or col < 0 or col >= len(grid[0]) or grid[row][col] == "0":
        return
    grid[row][col] = "0"
    dfs(grid, row + 1, col)
    dfs(grid, row - 1, col)
    dfs(grid, row, col + 1)
    dfs(grid, row, col - 1)


class MSTInterviewPrepSolutions(object):

    def maxLength(self, arr):

        n = len(arr)
        result = 0

        def hasDuplicate(s):
            return len(s) != len(set(s))

        def backtrack(current_string, index):
            nonlocal result
            result = max(result, len(current_string))
            for i in range(index, n):
                new_str = current_string + arr[i]
                if not hasDuplicate(new_str):
                    backtrack(new_str, i + 1)

        backtrack("", 0)

        return result

    def spiralMatrix(self, matrix):

        top = 0
        bottom = len(matrix) - 1
        left = 0
        right = len(matrix[0]) - 1
        direction = "right"
        result = []
        if not matrix:
            return result

        while top <= bottom and left <= right:
            if direction == "right":
                for i in range(left, right + 1):
                    result.append(matrix[top][i])
                top += 1
                direction = "down"
            elif direction == "down":
                for i in range(top, bottom + 1):
                    result.append(matrix[i][right])
                right -= 1
                direction = "left"
            elif direction == "left":
                for i in range(right, left - 1, -1):
                    result.append(matrix[bottom][i])
                bottom -= 1
                direction = "up"

            elif direction == "up":
                for i in range(bottom, top - 1, -1):
                    result.append(matrix[i][left])
                left -= 1
                direction = "right"

        return result

        def firstMissingPositive(self, nums):

            """
      :type nums: List[int]
      :rtype: int
      Basic idea:
      1. for any array whose length is l, the first missing positive must be in range [1,...,l+1], 
          so we only have to care about those elements in this range and remove the rest.
      2. we can use the array index as the hash to restore the frequency of each number within 
          the range [1,...,l+1] 
      """

            n = len(nums)
            for i in range(len(nums)):
                num = nums[i]
                if num < 0 or num >= n:
                    num = 0  # delete those useless elements

            for i in range(len(nums)):
                x = nums[i] % n  # use the index as the has to record the freq of each number
                nums[x] += n

            for i in range(1, len(nums)):
                num = nums[i]
                if num / n == 0:
                    return i

            return n

    def maxNetworkRank(self, n, roads):

        adj = [0] * n + 1

        for a, b in roads:
            adj[a] += 1
            adj[b] += 1

            max_rank = 0

            for a, b in roads:
                max_rank = max(max_rank, adj[a] + adj[b] - 1)
        return max_rank

    def maxNetworkRank2(self, n, roads):

        adjacent_list = {}

        for vertex in range(n):
            adjacent_list[vertex] = []

        for sub in roads:
            v1 = sub[0]
            v2 = sub[1]
            adjacent_list[v1].append(v2)
            adjacent_list[v2].append(v1)

        max_so_far = 0
        for i in range(n):
            for j in range(i + 1, n):
                max_so_far = max(max_so_far, len(adjacent_list[i]) + len(adjacent_list[j]) - (i in adjacent_list[j]))

        return max_so_far

    def mergeKLists(lists: List[ListNode]) -> ListNode:

        dummy = ListNode(0)
        output = dummy
        queue = []
        for head in lists:
            while head is not None:
                heappush(queue, head.val)
                head = head.next

        while len(queue) > 0:
            output.next = ListNode(heappop(queue))
            output = output.next

        return dummy.next

    def maxLength2(self, arr):

        result = 0

        def hasDup(s):
            return len(s) != len(set(s))

        def dfs(current_str, index):

            # nonlocal result

            # result = max(result,len(current_str))

            for i in range(index, len(arr)):
                new_str = current_str + arr[i]

                if not hasDup(new_str):
                    dfs(new_str, i + 1)

        dfs("", 0)
        return result

        # solution not clear

    def modifyString(self, s: str) -> str:
        s = list(s)
        for i in range(len(s)):
            if s[i] == "?":
                for c in "abc":
                    if (i == 0 or s[i - 1] != c) and (i + 1 == len(s) or s[i + 1] != c):
                        s[i] = c
                        break
        return "".join(s)


def reverseWords(s: str) -> int:
    result = []
    str_new = s.split(" ")
    print(str_new)

    for i in range(len(str_new) - 1, -1, -1):
        word = str_new[i]
        if len(word) > 0:
            result.append(word)

    return " ".join(result)


class QueueUsingStack():

    def __init__(self):
        self.stack1 = []
        self.stack2 = []

    def push(self, x):
        self.stack1.append(x)

    def pop(self):

        # fill the stack2 first and then pop the top
        self.peek()
        return self.stack2.pop()

    def peek(self):
        if not self.stack2:
            while self.stack1:
                self.stack2.append(self.stack1.pop())

        return self.stack2[-1]


class Queue(object):

    def __init__(self):
        self.queue = []

    def isEmpty(self):
        return len(self.queue) == 0

    def enqueue(self, item):
        self.queue.insert(0, item)  # add to the front

    def dequeue(self):
        return self.queue.pop()

    def size(self):
        return len(self.queue)


def partitionDisjoint(nums):
    n = len(nums)
    maxleft = [None] * n
    minright = [None] * n
    m = nums[0]
    for i in range(n):
        m = max(m, nums[i])
        maxleft[i] = m

    m = nums[-1]
    for i in range(n - 1, -1, -1):
        m = min(m, nums[i])
        minright[i] = m

    for i in range(1, n):
        if maxleft[i - 1] <= minright[i]:
            return i


def reverseWords2(s):
    def reverse_word_helper(left, right):
        while left < right:
            s[left], s[right] = s[right], s[right]
            left += 1
            right -= 1

        reverse_word_helper(0, len(s) - 1)

    left = 0
    for i, char in enumerate(s):
        if char == " ":
            reverse_word_helper(left, i - 1)
            left = i + 1
        reverse_word_helper(left, len(s) - 1)


"""
Dynamic Programming 

"""


class DPSolutions():

    def fibo(self, n):
        # 0 ,1, 1,2,3,5,8
        dp = [0, 1]
        for i in range(2, n + 1):
            dp.append(dp[i - 1] + dp[i - 2])
            # dp[i] = dp[i-1] + dp[i-2]

        return dp.pop()

    def climbingStairs(self, n: int) -> int:

        if n <= 2:
            return n

        dp = [0, 1, 2]

        for i in range(3, n + 1):
            dp.append(dp[i - 1] + dp[i - 2])

        return dp.pop()

    def houseRobber(self, nums: List[int]) -> int:
        # base case
        dp = [nums[0], max(nums[0], nums[1])]

        for i in range(2, len(nums)):
            dp.append(max(nums[i] + dp[i - 2], dp[i - 1]))

        print(dp)

        return dp.pop()

    def canJump(self, nums: List[int]) -> bool:
        dp = [False] * len(nums)
        dp[0] = True
        for j in range(1, len(nums)):
            for i in range(j):
                if dp[i] and i + nums[i] >= j:
                    dp[j] = True
        return dp.pop()

    def canJump2(self, nums: List[int]) -> bool:

        max_reach = 0
        for i in range(len((nums))):
            if i > max_reach:
                return False

            current_reach = i + nums[i]

            max_reach = max(max_reach, current_reach)

        return True

    def lengthOfLongestSubsequence(self, nums: List[int]) -> int:

        # [1,4,0,7]

        dp = [1] * len(nums)

        maxlen = 1

        for j in range(1, len((nums))):
            for i in range(j):
                if nums[j] > nums[i]:
                    dp[j] = max(dp[i] + 1, dp[j])

            maxlen = max(maxlen, dp[j])

        return maxlen

    def uniquePath(self, m: int, n: int) -> int:

        dp = [[1 for col in range(m)] for row in range(n)]

        for row in range(1, n):
            for col in range(1, m):
                dp[row][col] = dp[row][col - 1] + dp[row - 1][col]

        return dp[-1][-1]


"""
Two pointers questions : mostly used to solve iterative problems such as arrays 
"""


class Node:
    def __init__(self, val, next=None):
        self.val = val
        self.next = next


class TwoPointersSolutions:

    def removeDuplicateFromSortedArray(self, nums: List[int]) -> int:

        windowStart = 0
        for i in range(len(nums)):

            if nums[i] != nums[windowStart]:
                windowStart += 1

                nums[windowStart] = nums[i]
        return windowStart + 1

    def removeDupFromSortedArray2(self, nums: List[int]) -> int:
        windowStart = 0
        count = 0
        for i in range(len(nums)):
            if nums[windowStart] != nums[i]:
                windowStart = i
                count += 1
        return count + 1

    def findMiddleOfLinkedList(self, head: Node) -> int:

        slow, fast = head, head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next

        return slow.val

    def moveZeros(self, nums: List[int]):
        index = 0
        for i in range(len(nums)):
            if nums[i] != 0:
                nums[index] = nums[i]
                index += 1
        for i in range(index, len(nums)):
            nums[i] = 0

    def moveZeros2(self, nums: List[int]):

        window = 0
        for i in range(len(nums)):
            if nums[i] != 0:
                nums[window], nums[i] = nums[i], nums[window]
                window += 1

    def twoSumSortedArrays(self, nums: List[int], target: int) -> List[int]:
        result = []
        left = 0
        right = len(nums) - 1
        while left < right:
            temp = nums[left] + nums[right]
            if temp == target:
                result[0] = left
                result[1] = right

                left += 1
                right -= 1
            elif temp > target:
                right -= 1
            else:
                left += 1
        return result

    def longestSubStringWithoutRepeatingChars(self, string: str) -> int:
        pass

    def findAllAnagrams(self, s: str, p: str) -> List[int]:
        dic = collections.Counter(p)
        result = []
        left = 0
        windowStart = 0
        while left < len(p):

            for i in range(len(s)):
                char = s[i]
                if char in dic:
                    result.append(windowStart)

            left += 1

    def subarraySum(self, nums: List[int], target: int) -> int:

        current_sum = 0
        count = 0
        dic = collections.Counter()
        dic[0] = 1

        for i in range(len((nums))):
            current_sum += nums[i]
            temp = target - current_sum

            if temp in dic:
                count += dic[temp]
            dic[current_sum] += 1

        return count

    def sortedString(self, string: str) -> str:
        return collections.Counter(string)


"""
Arrays Solutions 
"""


class Interval(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end


class ListNode1:
    def __init__(self, val, next):
        self.next = next
        self.val = val


class ArraySolutions:

    def containsDuplicate(self, nums):
        nums.sort()
        print(nums)
        for i in range(1, len(nums)):
            if nums[i] == nums[i - 1]:
                return True

        return False

    def productArraysExceptSelf(self, nums: List[int]) -> List[int]:
        res = [1] * len(nums)
        mul = 1
        for i in range(len(nums) - 1, -1, -1):
            res[i] = mul * res[i]
            mul = mul * nums[i]

        mul = 1
        for i in range(len(nums)):
            res[i] = res[i] * mul
            mul = mul * nums[i]

        return res

    def bestTimeToBuyAndSell(self, nums: List[int]) -> int:
        cheapest_price = nums[0]
        profit = -inf

        for price in nums:
            if price < cheapest_price:
                cheapest_price = price
            currentProfit = price - cheapest_price
            profit = max(profit, currentProfit)
        return profit

    def maximumContigousSubArray(self, nums: List[int]) -> int:

        maxSum = -inf
        currentSum = nums[0]
        for i in range(1, len(nums)):
            currentSum = max(currentSum + nums[i], nums[i])
            maxSum = max(maxSum, currentSum)
        return maxSum


class LeetcodeBlindSolutions:
    def findMinimumInRotatedSortedArray(self, nums: List[int]) -> int:
        boundaryIndex = -1
        left = 0
        right = len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] <= nums[-1]:
                boundaryIndex = mid
                right = mid - 1
            else:
                left = mid + 1

        return nums[boundaryIndex]

    def searchInRotatedSortedArray(self, nums: List[int], target: int) -> int:
        left = 0
        right = len(nums)
        while left <= right:
            mid = (left + right) // 2

            if nums[mid] == target:
                return mid
            else:
                if nums[mid] > nums[left]:

                    if target <= nums[mid] and target >= nums[left]:

                        right = mid - 1
                    else:
                        left = mid + 1
                else:

                    if target >= nums[mid] and target <= nums[right]:
                        left = mid + 1
                    else:
                        rightr = mid - 1

        return -1

    def findMissingNumber(self, nums: List[int]) -> int:
        nums_set = set(nums)
        n = len(nums) + 1

        for i in range(n):
            if i not in nums_set:
                return i

    def longestContinousIncreasingSubsequence(self, nums: List[int]) -> int:
        windowStart = 0
        maxLen = -inf

        for i in range(0, len(nums)):

            if i > 0 and nums[i] < nums[i - 1]:
                continue
            else:
                maxlen = max(maxlen, i - windowStart + 1)
                windowStart = i
        return maxlen

    def isAnagram(self, s1: str, s2: str):
        if len(s1) != len(s2):
            return False

        check = [0] * 26

        for char in s1:
            ch = ord(char) - ord('a')
            check[ch] += 1
        for char in s2:
            ch = ord(char) - ord('a')
            check[ch] -= 1

        for num in check:
            if num > 0:
                return False
        return True

    def permutations(self, string: str) -> List[str]:

        def dfs(used: bool, path: List, res: List[str]):
            if len(path) == len(string):
                res.append(''.join(path))
                return

            for i, c in enumerate(string):
                if used[i]:
                    continue
                path.append(c)
                used[i] = True
                dfs(used, path, res)

                path.pop()
                used[i] = False

        res = []
        dfs([] * len(string), [], res)

        return res

    def combinationSum(self, nums: List[int], target: int) -> List[List[int]]:
        res = []

        def dfs(i, path, total):
            if total == target:
                path.append()
                return

    def mergeInterval(self, intervals):
        # Input: intervals = [[1,3],[2,6],[8,10],[15,18]]
        # Output: [[1,6],[8,10],[15,18]]

        result = [intervals[0]]

        for interval in intervals:
            start = interval[0]
            end = interval[1]
            lastEnd = result[len(result) - 1][1]

            # check for overlap

            if start <= lastEnd:
                result[len(result) - 1][1] = max(end, lastEnd)
            else:
                result.append(interval)
        return result

    def eraseOverlapIntervals(self, intervals: List[List[int]]):
        # Input: intervals = [[1,2],[2,3]]
        # Output: 0
        intervals.sort(key=lambda i: i[0])
        result = [intervals[0]]
        res = 0

        for interval in intervals[1:]:
            start = interval[0]
            end = interval[1]
            lastEnd = result[-1][1]
            if start < lastEnd:
                res += 1
                result[-1][1] = max(lastEnd, end)
            else:
                result.append(interval)
        return res

    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:

        dummy = ListNode()
        res = dummy

        while l1 and l2:
            if l1.val < l2.val:
                dummy.next = l1
                l1 = l1.next
            else:
                dummy.next = l2
                l2 = l2.next
        if l1:
            dummy.next = l1
        elif l2:
            dummy.next = l2
        return res.next

    def deleteDuplicates(self, head: ListNode) -> ListNode:
        current = head

        while current:
            while current.next and current.next.val == current.val:
                current.next = current.next.next

            current = current.next

        return head

    def removeElements(self, head: ListNode, val: int) -> ListNode:
        while head != None and head.val == val:
            head = head.next

        current = head
        while current != None and current.next != None:
            if current.next.val == val:
                current.next = current.next.next
            else:
                current = current.next
        return head

    def longestPalindrome(self, string: str) -> int:
        data = set()
        count = 0

        for char in string:
            if char in data:
                count += 1
                data.pop()
            else:
                data.add(char)

        count = count * 2

        if len(data) >= 1:
            count += 1

        return count

    def longestOnes(self, nums: List[int], k: int):
        windowStart = 0
        maxlen = 0

        left = 0

        while left < len(nums):

            if nums[left] == 0:
                k -= 1
            if k < 0:
                if nums[windowStart] == 0:
                    k += 1

                windowStart += 1

            left += 1

        return left - windowStart

    def pivotIndex(self, nums: List[int]):
        summ = sum(nums)

        accumSum = 0

        for i in range(len(nums)):
            if summ - accumSum - nums[i] == accumSum:
                return i
            accumSum += nums[i]

        return -1

    def rotateArray(self, nums: List[int], k: int):
        k = k % len(nums)
        left = 0
        right = len(nums) - 1
        self.helper(nums, left, right)
        self.helper(nums, 0, k - 1)
        self.helper(nums, k, right)

    def helper(self, nums, left, right):
        while left < right:
            nums[left], nums[right] = nums[right], nums[left]

            left += 1
            right -= 1

    def maxConsecutiveOnes(self, nums: List[int]):
        maxLen = -inf
        windowStart = 0

        for i in range(len(nums)):

            if nums[i] == 0:
                maxLen = max(i - windowStart + 1, maxLen)
                windowStart = i + 1
            else:
                continue

        return maxLen

    def removeAllAdjancentDuplicates(self, string: str):
        stack = []
        res = []
        for char in string:
            if len(stack) == 0 or stack[-1] != char:
                stack.append(char)
            else:
                stack.pop()

        while len(stack) > 0:
            res.append(stack.pop())

        return ''.join(res)

    def searchMatrix(self, grid, target):
        pass

    def isPalindromeLinkedList(self, head: ListNode):
        res = []
        while head != None:
            res.append(head.val)
            head = head.next
        left = 0
        right = len(res) - 1
        while left < right:
            if res[left] != res[right]:
                return False
            left += 1
            right -= 1
        return True

    def palindromeNumber(self, num):
        if num < 0:
            return False
        string = str(abs(num))

        left = 0
        right = len(string) - 1
        while left < right:
            if string[left] != string[right]:
                return False
            left += 1
            right -= 1

        return True

    def linkedListCycle(self, head: ListNode):
        slow = head
        fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                return True
        return True

    def numIslands(self, grid):

        count = 0
        for row in range(len(grid)):
            for col in range(len(grid[row])):
                if grid[row][col] == '1':
                    count += 1
                    self.dfs(grid, row, col)

    def dfs(self, grid, row, col):
        if row < 0 or row >= len(grid) or col < 0 or col >= len(grid[row] or grid[row][col] == 0):
            return

        grid[row][col] = '0'
        self.dfs(grid, row + 1, col)
        self.dfs(grid, row - 1, col)
        self.dfs(grid, row, col + 1)
        self.dfs(grid, row, col - 1)

    def middleNode(self, head: ListNode):
        if head == None:
            return head

        slow = head
        fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next

        return slow

    def mergeTwoSortedLists(self, l1: ListNode, l2: ListNode):

        dummy = ListNode()
        res = dummy

        while l1 or l2:
            if l1 is not None:
                if l1.val < l2.val:
                    dummy.next = l1
                    l1 = l1.next
            else:
                dummy.next = l2
                l2 = l2.next

        if l1:
            dummy.next = l1
        if l2:
            dummy.next = l2

        return res

    def addTwoNumbers(self, l1: ListNode, l2: ListNode):
        carry = 0
        dummy = ListNode()
        res = dummy

        while l1 or l2 or carry != 0:
            if l1:
                carry += l1.val
                l1 = l1.next
            if l2:
                carry += l2.val
                l2 = l2.next

            dummy.next = ListNode(carry % 10)
            carry = carry / 10

            dummy = dummy.next

        return res.next

    def minSubArrayLen(self, nums: List[int], s: int):
        windowStart = 0
        minLen = inf
        summation = 0

        for i in range(len(nums)):
            summation += nums[i]

            while summation >= s:
                minLen = min(minLen, i - windowStart + 1)
                summation -= nums[windowStart]
                windowStart += 1

        if minLen == inf:
            return 0
        return minLen

    def groupAnagrams(self, strs: List[str]):
        results = []
        dic = {}

        for string in strs:
            hashed = self.hashedString(string)
            if hashed not in dic:
                dic[hashed] = []
                dic[hashed].append(string)

        for p in dic.values():
            results.append(p)

        return results

    def hashedString(self, s):
        return ''.join(sorted(s))

    def reverseString(self, s):

        res = []
        for i in range(len(s) - 1, -1, -1):
            res.append(s[i])

        return ''.join(res)

    def findPeakElement(self, nums: List[int]):
        left = 0
        right = len(nums) - 1
        while left < right:
            mid = left + (right - left) // 2

            if nums[mid] < nums[mid + 1]:
                left = mid + 1
            else:
                right = mid
        return left

    def frequencySort(self, s: str):
        dic = collections.Counter(s)
        s = list(s)

        s.sort(key=lambda x: (-dic[x], x))
        print(s)

        return ''.join(s)

    def addString(self, num1: str, num2: str):

        res = []
        carry = 0
        l1 = len(num1) - 1
        l2 = len(num2) - 1
        string = ""
        while l1 >= 0 or l2 >= 0:
            summ = carry
            if l1 >= 0:
                summ += ord(num1[l1]) - ord('0')
            if l2 >= 0:
                summ += ord(num2[l2]) - ord('0')
            res.append(summ % 10)
            carry = summ // 10

            l1 -= 1
            l2 -= 1
        if carry != 0:
            res.append(carry)

        for i in range(len(res) - 1, -1, -1):
            string += str(res[i])
        print(string)

        return ''.join(str(x) for x in res[::-1])

    def encodeString(self, strs: List[str]):

        encoded = ""

        for string in strs:
            length = len(string)

            encoded += str(length) + "/" + string
        return encoded

    def decodeString(self, string: str):
        # 5/kevin2/is5/great

        pos = 0
        decoded = []

        while pos < len(string):
            slash_pos = string.index("/", pos)
            pos = slash_pos + 1

            length = int(slash_pos - 1)
            decoded.append(string[pos:pos + length])

            pos += length

        return decoded

    def uniquePath(self, m, n):
        dp = [[1 for col in range(m)] for row in range(n)]

        for row in range(1, n):
            for col in range(1, m):
                dp[row][col] = dp[row][col - 1] + dp[row - 1][col]

        return dp[-1][-1]

    def jumpGame(self, nums):
        last = len(nums) - 1
        for i in range(len(nums) - 1, -1, -1):
            if i + nums[i] >= last:
                last = i

        return last == 0

    def spiralMatrix(self, grid):
        result = []

        if not grid:
            return result

        top = 0
        bottom = len(grid) - 1
        left = 0
        right = len(grid[0]) - 1
        direction = "right"
        while top <= bottom and left <= right:
            if direction == "right":
                for i in range(left, right + 1):
                    result.append(grid[top][i])
                top += 1
                direction = "down"
            elif direction == "down":
                for i in range(top, bottom + 1):
                    result.append(grid[i][right])
                right -= 1
                direction = "left"
            elif direction == "left":
                for i in range(right, left - 1, -1):
                    result.append(grid[bottom][i])
                bottom -= 1
                direction = "up"
            elif direction == "up":
                for i in range(bottom, top - 1, -1):
                    result.append(grid[i][left])
                left += 1
                direction = "right"
        return result

    def wordSearch(self, grid, word):
        found = False
        for row in range(0, len(grid)):
            for col in range(0, len(grid[0])):
                if grid[row][col] == word[0]:
                    self.dfs_word_search(grid, row, col, 0, word)

        return found

    def dfs_word_search(self, grid, row, col, count, word):
        if count == len(word):
            nonlocal found
            found = True
            return

        if row < 0 or row >= len(grid) or col < 0 or col >= len(grid[0]) or grid[row][col] != word[count] or found:
            return

        temp = grid[row][col]
        grid[row][col] = ""
        self.dfs_word_search(grid, row + 1, col, count + 1, word)
        self.dfs_word_search(grid, row + 1, col, count + 1, word)
        self.dfs_word_search(grid, row + 1, col, count + 1, word)
        self.dfs_word_search(grid, row + 1, col, count + 1, word)
        grid[row][col] = temp

    def reverseLinklist(self, head: ListNode):
        output = None
        while head != None:
            temp = head.next
            head.next = output
            output = head
            head = temp

        return output


class Queue:

    def __init__(self):

        self.push_stack = []
        self.pop_stack = []

    def push(self, x):
        self.push_stack.append(x)

    def peek(self):

        return self.pop_stack.pop()

    def peek(self):
        if not self.pop_stack:
            while self.push_stack:
                self.pop_stack.append(self.push_stack.pop())

        return self.pop_stack[-1]

    def empty(self):
        return not self.pop_stack and not self.push_stack


def firstUniqChar(s: str):
    dic = dict()
    for char in s:
        if char in dic:
            dic[char] += 1
        else:
            dic[char] = 1

    for i in range(len(s)):
        if dic[s[i]] == 1:
            return i
    return -1


if __name__ == "__main__":
    print(LeetcodeBlindSolutions.addString("123", "456"))
