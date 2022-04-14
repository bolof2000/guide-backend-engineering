from collections import defaultdict

from leetcode.listnode import ListNode
from leetcode.questions import TreeNode


def groupAnagram(strs):
    dic = defaultdict(int)
    for word in strs:
        hashed = hashString(word)
        if hashed not in dic:
            dic[hashed] = []
        dic[hashed].append(word)

    return list(dic.values())


def hashString(s):
    return "".join(sorted(s))


def uniquePath(m: int, n: int) -> int:
    dp_matrix = [[1 for i in range(m)] for i in range(n)]

    for row in range(1, m):
        for col in range(1, n):
            dp_matrix[row][col] = dp_matrix[row][col - 1] + dp_matrix[row - 1][col]

    return dp_matrix[-1][-1]


def shortestDistance(wordDict, word1, word2):
    result = first = second = float('inf')

    for i, word in enumerate(wordDict):
        if word == word1:
            first = i
        elif word == word2:
            second == i
        result = min(abs(first - second), result)
    return result


""" 
tiny url solutions
"""


class Codec:

    def __init__(self):
        self.encoded = {}
        self.decoded = {}
        self.base = "https://tinyurl.com"

    def encode(self, longUrl):
        if longUrl not in self.encoded:
            shortUrl = self.base + str(len(self.encoded) + 1)
            self.encoded[longUrl] = shortUrl
            self.decoded[shortUrl] = longUrl

        return self.encoded[longUrl]

    def decode(self, shortUrl):
        return self.decoded[shortUrl]


def subarraySum(nums, k):
    dic = defaultdict(int)

    dic.setdefault(0, 1)

    count = 0
    summ = 0
    for num in nums:
        summ += num
        temp = summ - k

        if temp in dic:
            count += dic.get(temp)

        dic[summ] += 1
    return count


def hasCycle(self, head):
    """
    :type head: ListNode
    :rtype: bool
    """
    node_set = set()
    current = head
    while current:
        if current in node_set:
            return True
        else:
            node_set.add(current)
            current = current.next

    return False


def middleNode(head: ListNode) -> ListNode:
    if not head:
        return head
    slow = head
    fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    return slow


def mergeTwoLists(l1: ListNode, l2: ListNode):
    dummy = ListNode(0)
    output = dummy

    while l1 is not None and l2 is not None:

        if l1.val > l2.val:
            dummy.next = l2.next
            l2 = l2.next

        else:

            dummy.next = l1.next
            l1 = l1.next

        dummy = dummy.next

    if l1 is not None:
        dummy.next = l1
    if l2 is not None:
        dummy.next = l2

    return output.next


def removeNthFromEnd(head: ListNode, n: int) -> ListNode:
    dummy = ListNode(0)
    output = dummy
    fast = head
    slow = head
    while n > 0:
        fast = fast.next
        n -= 1

    while fast and fast.next:
        slow = slow.next
        fast = fast.next

    slow.next = slow.next.next

    return dummy.next


def lca(root,node1,node2):
    if not root:
        return

    if root == node1 or root == node2:

        return root

    left = lca(root.left,node1,node2)
    right = lca(root.right,node1,node2)
    if left and right:
        return root
    if left:
        return left
    if right:
        return right
    return None

def good_nodes(root:TreeNode)->int:

    def dfs(root,max_so_far):

        if not root:
            return 0
        total = 0
        if root.val >= max_so_far:
            total +=1
        total += dfs(root.left,max(max_so_far,root.val))
        total += dfs(root.right,max(max_so_far,root.val))
        return total

    dfs(root, -float('inf'))

def permute(s):

    res = []
    def dfs(path,used,res):
        if len(path) ==len(s):
            res.append(path[:])
            return
        for i,letter in enumerate(s):
            if used[i]:
                continue
            path.append(letter)
            used[i] = True
            dfs(path,used,res)
            path.pop()
            used[i] = False

        dfs([],[False]*len(s),res)
        return res

def wordBreak(s,worddict):
    def dfs(i,memo):
        if i == len(s):
            return True
        if i in memo:
            return memo[i]
        ok = False

        for word in worddict:
            if s[i:].startswith(word):
                if dfs(i+len(word),memo):
                    ok = True
                    break

        memo[i] = ok
        return ok

    return dfs(0,{})

def decode_ways(digits):
    prefixes = [str(i) for i in range(1,27)]

    def dfs(i,memo):
        if i == len(digits):
            return 1
        if i in memo:
            return memo[i]
        ways = 0

        for prefix in prefixes:
            if digits[i:].startswith(prefix):
                ways += dfs(i+len(prefix),memo)

        memo[i] = ways

        return ways

    return dfs(0,{})

def combination_sum(candidates,target):

    res = []

    def dfs(nums,start_index,remaining,path):
        if remaining ==0:
            res.append(path[:])
            return
        for i in range(start_index,len(nums)):
            num = nums[i]
            if remaining-num <0:
                continue
            dfs(nums,i,remaining-num,path+[num])

    dfs(candidates,0,target,[])
    return res


