# View more python learning tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

char_list = ['a', 'b', 'c', 'c', 'd', 'd', 'd']

sentence = 'Welcome Back to This Tutorial'

print(set(char_list))
print(set(sentence))

print(set(char_list + list(sentence)))

unique_char = set(char_list)
unique_char.add('x')
# unique_char.add(['y', 'z']) this is wrong
print(unique_char)

unique_char.remove('x')
print(unique_char)
unique_char.discard('d')
print(unique_char)
unique_char.clear()
print(unique_char)

unique_char = set(char_list)
print(unique_char.difference({'a', 'e', 'i'}))
print(unique_char.intersection({'a', 'e', 'i'}))