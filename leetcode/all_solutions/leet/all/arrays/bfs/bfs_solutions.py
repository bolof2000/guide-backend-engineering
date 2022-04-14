from collections import deque


class BFSSolutions:
    """

    TO solve BFS questions, A deque is used to keep track of the of the nodes
    the node is then popped
    the node value is then store in a list
    the loops then check for the childs node for both left and right
    the node of the child is added to the queue if not none

    """

    def level_order_tranversal(self, Node):

        queue = deque([Node])
        res = []

        while len(queue) > 0:

            n = len(queue)
            new_level = []

            for _ in range(n):

                node = queue.popleft()  # deque each node in the current level

                new_level.append(node.val)

                for child in (Node.left, Node.right):  # enqueue non-null children
                    if child is not None:
                        queue.append(child)

            res.append(new_level)

        return res

    def zig_zag_transversal(self, Node):

        res = []
        queue = deque([Node])
        flag = True
        while len(queue) > 0:
            n = len(queue)
            next_levels = []

            for _ in range(n):
                node = queue.popleft()
                next_levels.append(node.val)

                for child in [Node.left, Node.right]:
                    if child is not None:
                        queue.append(child)

            if not flag:
                res.append(next_levels.reverse())

            flag = not flag

        return res

    def binary_tree_right_view(self,Node):

        queue = deque([Node])

        res = []

        while len(queue) > 0:

            n = len(queue)

            new_level = []

            node = queue.popleft()

            new_level.append(node.val)

            for child in Node.left:
                if child is not None:
                    queue.append(child)


            res.append(new_level)

        return res
