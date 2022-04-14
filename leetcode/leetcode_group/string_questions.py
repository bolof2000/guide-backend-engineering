from typing import List
def group_anagrams(strs:List[str]):
    results = []
    dic = {}
    for string in strs:
        key = hashed(string)
        if key not in dic:
            dic[key] = []
        dic[key].append(string)

    for item in dic.values():
        results.append(item)
    return results

def hashed(string):
    return "".join(sorted(string))

def valid_parenthesis(string):
    opening = set('[({')
    matches = set([('(',')'),('{','}'),('{','}')])
    stack = []
    for char in string:
        if char in opening:
            stack.append(char)
        else:
            if len(stack) ==0:
                return False
            if stack:
                open = stack.pop()
                if (open,char) in matches:
                    return True

    return len(stack) ==0

if __name__ == '__main__':
    strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
    #print(group_anagrams(strs))
    print(valid_parenthesis("}"))
