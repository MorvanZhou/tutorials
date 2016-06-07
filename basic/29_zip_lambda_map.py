a = [1,2,3]
b = [4,5,6]

# for zip
list(zip(a,b))
list(zip(a,a,b))
for i, j in zip(a,b):
    print(i,j)

#for lambda
def f1(x,y):
    return x+y
f2= lambda x, y : x + y
print(f1(1,2))
print(f2(1,2))

# for map
print(list(map(f1, [1],[2])))
print(list(map(f2, [2,3],[4,5])))
