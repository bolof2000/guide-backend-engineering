nums = [1,2,3,4,5]
for i in range(len(nums)-1,-1,-1):
    print(nums[i])

string = "abab1"
print(str.isalpha(string))
print(string.isdigit())

for char in string:
    if char.isdigit():
        print(char)