class TreeNode:
    def __init__(self,val=0,left=None,right=None):
        self.val = val
        self.right = right
        self.left = left

class DFSSolutions:

    def isSameTree(self,p:TreeNode,q:TreeNode)->bool:


        def dfs(left:TreeNode,right:TreeNode):
            if left is None and right is None:
                return True

            if left is None or right is None or left.val != right.val:
                return False

            return dfs(left.left,right.left) and dfs(left.right,right.right)

        return dfs(p,q)


    def mergeTrees(self, root1: TreeNode, root2: TreeNode) -> TreeNode:

        def dfs(root1:TreeNode,root2:TreeNode):
            if root1 is None:
                return root2
            if root2 is None:
                return root1
            result = TreeNode(root1.val+root2.val)
            result.left = dfs(root1.left,root2.left)
            result.right = dfs(root1.right,root2.right)
            return result
        return dfs(root1,root2)

    def invertTree(self, root: TreeNode) -> TreeNode:

        def dfs(root:TreeNode):
            if root is None:
                return
            temp = TreeNode()
            temp = root.left
            root.left = root.right
            root.right = temp
            dfs(root.left)
            dfs(root.right)

            return root

        return dfs(root)


