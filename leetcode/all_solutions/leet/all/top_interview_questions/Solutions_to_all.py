"""
Contains solutions to 145 most frequently asked questions.
Approach here is to solve the questions and all related questions to the main questions

if you solve two sum, then you should solve 3 sum, four sum since this solutions will be related
if you implemented a queue then you should implement a stack, deque and so on


"""
from collections import defaultdict
import pdb


class Solutions:

    def findMedianSortedArrays(self, nums1, nums2):
        pass

    def lengthOfLongestSubstringTwoDistinct(self, s):
        # Input: s = "ccaabbb"

        mapp = {}
        window_start = 0
        right = 0
        maxlen = 0
        # s = list(s)

        while right < len(s):

            mapp[s[right]] = right
            print(mapp)
            if len(mapp) == 3:
                vals = list(mapp.values())
                # print(min(vals))
                minVal = min(vals)
                print(s[minVal])

                print(minVal)
                del mapp[s[minVal]]

                window_start = minVal + 1
            maxlen = max(maxlen, right - window_start + 1)

            right += 1

        return maxlen

    def lengthOfLongestSubstring(self, s):

        # Input: s = "abcabcbb"

        window_start = 0
        right = 0
        maxlen = 0
        mapp = {}

        for i in range(len(s)):
            char = s[i]
            if char in mapp and mapp[char] >= window_start:
                pdb.set_trace()
                window_start = mapp[char] + 1

            mapp[char] = i

            maxlen = max(maxlen, i - window_start + 1)
        return maxlen

    def lengthOfLongestSubstringKDistinct(self, s, k):

        mapp = defaultdict(int)
        right = 0
        window_start = 0
        maxlen = 0

        for i in range(len(s)):
            char = s[i]
            if len(mapp) >= k:
                minval = min(mapp.values())
                window_start = minval + 1
            # continue

            mapp[char] = i
            print(mapp)

            maxlen = max(maxlen, i - window_start + 1)
        return maxlen

    def findMedianSortedArrays(self, nums1, nums2):

        mergedArray = []

        l1 = len(nums1)
        l2 = len(nums2)

        i = 0
        j = 0
        while i < l1 and j < l2:
            if nums1[i] <= nums2[j]:
                mergedArray.append(nums1[i])

                i += 1
            else:
                mergedArray.append(nums2[j])
                j += 1

        while i < l1:
            mergedArray.append(nums1[i])
            i += 1

        while j < l2:
            mergedArray.append(nums2[j])
            j +=1

        mergeLen = len(mergedArray)
        print(mergedArray)
        if len(mergedArray) % 2 != 0:
            return mergedArray[(mergeLen // 2)]
        else:
            return (mergedArray[mergeLen // 2] + mergedArray[(mergeLen //2) -1]) / 2


    def longestCommonPrefix(self, strs):

        if len(strs) == 0:
            return ""
        prefix = strs[0]
        prefix_len = len(prefix)
        for i in range(1,len(strs)):
            word = strs[i]
            while prefix != word[0:prefix_len]:
                prefix = prefix[0:prefix_len-1]  # decrease the slice
                prefix_len -=1
                if prefix_len ==0:
                    return ""
        return prefix






# Input: strs = ["flower","flow","flight"]


# Input: nums1 = [1,2], nums2 = [3,4]
# Output: 2.50000
# Explanation: merged array = [1,2,3,4] and median is (2 + 3) / 2 = 2.5.


test = Solutions()
# print(test.lengthOfLongestSubstringTwoDistinct("eceba"))
# print(test.lengthOfLongestSubstring("abcabcbb"))
#print(test.lengthOfLongestSubstringKDistinct("eceba", 2))
arr1 = [1,2]
arr2 = [3,4]
nums1 = [0,0]
nums2 = [0,0]
print(test.findMedianSortedArrays(arr1,arr2))
