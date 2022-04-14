from leetcode.listnode import ListNode


class ListNode:

    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

    def rotateRight(self, head: ListNode, k: int) -> ListNode:

        if not head:
            return head

        length, tail = 1, head
        while tail.next:
            tail = tail.next

            length += 1

        k = k % length
        if k == 0:
            return head

        current = head
        for i in range(length - k - 1):
            current = current.next

        newHead = current.next
        current.next = None
        tail.next = head

        return newHead
