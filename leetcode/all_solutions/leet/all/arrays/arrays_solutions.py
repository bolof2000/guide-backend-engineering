def maxProduct(self, nums):
    if len(nums) == 0:
        return 0

    result = nums[0]
    for i in range(len(nums)):
        accum =1
        for j in range(i,len(nums)):

            accum *= nums[j]

            result = max(accum,result)
    return result

def maxProduct2(nums):

    max = nums[0]
    min = nums[0]
    output = max



