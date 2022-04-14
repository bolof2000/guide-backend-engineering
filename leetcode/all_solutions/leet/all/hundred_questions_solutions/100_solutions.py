from collections import deque
from collections import defaultdict


class TreeNode(object):

    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class ListNode:

    def __init__(self, val=0, next=None):
        self.val = val,
        self.next = next


class Solutions(object):

    def isSymmetric(self, root):
        """
             Identify your return value
             Identify your state
             pass the state to the child nodes
       """

        def dfs(left, right):
            if left is None and right is None:
                return True

            if left is None or right is None or left.val != right.val:
                return False

            return dfs(left.left, right.right) and dfs(left.right, right.left)

        return dfs(root.left, root.right)

    def invertTree(self, root):

        def dfs(root):
            if not root:
                return

            temp = root.left
            root.left = root.right
            root.right = temp

            dfs(root.left)
            dfs(root.right)

        dfs(root)
        return root

    def mergeTrees(self, root1, root2):

        def dfs(t1, t2):

            if t1 is None:
                return t2

            if t2 is None:
                return t1

            if t1 is None and t2 is None:
                return
            val = t1.val + t2.val
            t3 = TreeNode(val)
            t3.left = dfs(t1.left, t2.left)
            t3.right = dfs(t2.right, t2.left)

            return t3

        return dfs(root1, root2)

    def twoSum(self, nums, target):
        mapp = defaultdict(int)
        res = []
        for i, num in enumerate(nums):
            temp = target - num

            if temp in mapp:
                res = [i, mapp[temp]]
            mapp[num] = i

        return res

    def containsDuplicate(self, nums):

        myset = set()
        for num in nums:
            myset.add(num)

        return len(myset) == len(nums)

    def reverseList(self, head):
        pass  # still do not understand this solution

    def twoSumInputArrayIsSorted(self, nums, target):

        left = 0
        res = []
        right = len(nums) - 1
        while left < right:
            temp = nums[left] + nums[right]
            if temp == target:

                res = [nums[left], nums[right]]

                left += 1
                right -= 1

            elif temp > target:
                right -= 1

            else:
                left -= 1

        return res

    def palindromeListList(self, head):

        res = []
        current = head
        while current is not None:
            res.append(head.val)

            current = current.next


        """
        Quick solutions:
         return res == res[::-1]
        
        """

        left = 0
        right = len(res) - 1
        while left < right:
            if res[left] != res[right]:
                return False
            left +=1
            right -=1

        return True


test = Solutions()
nums = [1, 2, 3, 4]
print(test.containsDuplicate(nums))
