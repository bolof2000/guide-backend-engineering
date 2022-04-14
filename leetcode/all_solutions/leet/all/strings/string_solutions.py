from collections import defaultdict
from collections import Counter
import pdb


class Solutions:

    def groupAnagrams(self, strs):

        mapp = {}
        res = []
        for str in strs:
            hashed = self.hashedStr(str)
            if hashed not in mapp:
                mapp[hashed] = []

            mapp[hashed].append(str)

        for st in mapp.values():
            res.append(st)

        return st

    def hashedStr(self, s):
        return ''.join(reversed(s))

    def isAnagram(self, s, t):
        if len(s) != len(t):
            return False

        mapp = defaultdict(int)

        for char in s:
            if char in mapp:
                mapp[char] += 1
            else:
                mapp[char] = 1

        for char in t:
            if char in mapp:
                mapp[char] -= 1
            else:
                mapp[char] = 1

        for k in mapp:
            if mapp[k] != 0:
                return False
        return True

    def encode(self, strs):

        pass

    def decode(self, s):
        pass

    def longestPalindromeSub(self, s):

        res = ""
        for i in range(len(s)):
            current = self.expandArroundTheMiddle(s, i - 1, i + 1)
            in_btw = self.expandArroundTheMiddle(s, i, i + 1)

            res = max(current, in_btw, res, key=len)

        return res

    def expandArroundTheMiddle(self, s, left, right):

        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1

        return s[left + 1, right]

    def longestPalindromeSub(self, s):

        longest = ""
        for i in range(len(s)):
            for j in range(i, len(s)):
                substring = s[i:j + 1]
                if len(substring) > len(longest) and self.isPalindrome(substring):
                    longest = substring

        return longest

    def isPalindrome(self, s):
        left = 0
        right = len(s) - 1
        while left < right:
            if s[left] != s[right]:
                return False

            left += 1
            right -= 1

        return True

    @staticmethod
    def lengthOfLongestSub(s):
        mapp = defaultdict(int)
        windows_start = 0
        maxLen = 0
        for i in range(len(s)):
            char = s[i]
            if char in mapp and mapp[char] >= windows_start:
                windows_start = mapp[char] + 1
            mapp[char] = i
            maxLen = max(maxLen, i - windows_start + 1)
        return maxLen

    @staticmethod
    def validParenthesis(s):
        stack = []

        for char in s:
            if char == '(' or char == '{' or char == '[':
                stack.append(char)
            elif char == ')' and len(stack) > 0 and stack[-1] == '(':
                stack.pop()
            elif char == '}' and len(stack) > 0 and stack[-1] == '{':
                stack.pop()
            elif char == ']' and len(stack) > 0 and stack[-1] == '[':
                stack.pop()
            else:
                return False

        return len(stack) == 0

    @staticmethod
    def backspaceString(s, t):
        stack1 = []
        stack2 = []

        for char in s:
            if char == '#' and len(stack1) > 0:
                stack1.pop()
            else:
                stack1.append(char)

        for char in t:
            if char == '#' and len(stack2) > 0:
                stack2.pop()
            else:
                stack2.append(char)
        return stack1 == stack2

    @staticmethod
    def decodeString(s):
        stack = []
        for char in s:
            if char != ']':
                stack.append(char)
            else:
                res = []
                while len(stack) > 0 and stack[-1].isalpha():
                    res.insert(0, stack.pop())

                finalS = "".join(res)
                print(finalS)

                stack.pop()

                digits = []
                while len(stack) > 0 and stack[-1].isdigit():
                    digits.insert(0, stack.pop())

                digits_join = "".join(digits)
                print(digits_join)
                count = int(digits_join)

                while count > 0:
                    for i in finalS:
                        stack.append(i)

                    count -= 1

        result = []
        while len(stack) > 0:
            result.insert(0, stack.pop())

        return "".join(result)

    @staticmethod
    def findAnagrams(s, p):
        p_len = len(p)
        s_len = len(s)

        if s_len < p_len:
            return []

        p_counter = Counter(p)
        s_counter = Counter()

        result = []

        for i in range(s_len):
            char = s[i]
            s_counter[char] += 1
            #print(s_counter)
            #print(p_counter)

            #pdb.set_trace()

            if i >= p_len:
                print(s[i-p_len])
                if s_counter[s[i - p_len]] == 1:
                    del s_counter[s[i - p_len]]
                else:
                    s_counter[s[i - p_len]] -= 1
            if p_counter == s_counter:
                result.append(i - p_len + 1)
        return result


test = Solutions()
# print(test.longestPalindromeSub("babad"))
# print(test.lengthOfLongestSub("abcdabcdeabacda"))
# print(Solutions.validParenthesis("[[[[[[(){}[]]]]]]]]"))
# print(Solutions.backspaceString("a##c", "#a#c"))
# print(Solutions.decodeString("3[a]2[bc]"))
s = "cbaebabacd"
p = "abc"

print(Solutions.findAnagrams(s, p))
