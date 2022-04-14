from math import inf
class DFSSolutions:
    """

    When to use DFS :

    When to use DFS
Tree
DFS is essentially pre-order tree traversal.

Traverse and find/create/modify/delete node
Traverse with return value (finding max subtree, detect balanced tree)
Combinatorial problems
DFS/backtracking and combinatorial problems are a match made in heaven (or silver bullet and werewolf 😅). As we will see in the Combinatorial Search module, combinatorial search problems boil down to searching in trees.

DFS on Trees

Think like a Node

Defining the recursive function
Two things we need to decide to define the function:

1. Return value (passing value up from child to parent)
What do we want to return after visiting a node. For example, for max depth problem this is max depth for the current node's subtree. If we are looking for a node in the tree, we'd want to return that node if found, else return null. Use return value to pass information from children to parent.

2. Identify state(s) (passing value down from parent to child)
What states do we need to maintain to compute the return value for the current node. For example, to know if the current node's value is larger than its parent we have to maintain the parent's value as a state. State becomes DFS's function arguments. Use states to pass information from parent to children.




    """

    def maxDepth(self, Node):

        def dfs(root):
            if not root:
                return 0

            return max(dfs(Node.left), dfs(Node.right)) + 1

        return dfs(Node)

    def countVisibleNodes(self, Node):

        def dfs(Node, max_so_far):

            total = 0
            if not Node:
                return 0

            if Node.val >= max_so_far:
                total += 1

            total += dfs(Node.left, max(max_so_far, Node.val))
            total += dfs(Node.right, max(max_so_far, Node.val))
            return total

        return dfs(Node, -float('inf'))

    def validateBST(self, Node):

        def dfs(Node, minVal, maxVal):
            if not Node:
                return True

            if minVal >= Node.val >= maxVal:
                return False

            return dfs(Node.left,minVal,Node.val) and dfs(Node.right,Node.val,maxVal)

        return dfs(Node,-inf,inf)
