class BSTSolutions:

    def mySqrt(self, x):
        """
        :type x: int
        :rtype: int
        """
        left = 0
        right = x

        while left <= right:
            midpoint = (left + right) // 2

            product = int(midpoint * midpoint)
            if product > x:
                right = midpoint - 1
            elif product < x:
                left = midpoint + 1
            else:
                return midpoint

        return right


test = BSTSolutions()
print(test.mySqrt(4))
