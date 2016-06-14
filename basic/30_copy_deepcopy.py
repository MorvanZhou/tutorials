# View more python learning tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

import copy

a = [1,2,3]
b = a
b[1]=22
print(a)
print(id(a) == id(b))

# deep copy
c = copy.deepcopy(a)
print(id(a) == id(c))
c[1] = 2
print(a)
a[1] = 111
print(c)

# shallow copy
a = [1,2,[3,4]]
d = copy.copy(a)
print(id(a) == id(d))
print(id(a[2]) == id(d[2]))
