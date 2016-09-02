# View more python learning tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

char_list = ['a', 'b', 'c', 'c', 'd', 'd', 'd']

sentence = 'Welcome Back to This Tutorial'

print(set(char_list))
print(set(sentence))

print(set([char_list, sentence]))

unique_char = set(char_list)
unique_char.add('x')
unique_char.add(['y', 'z'])
print(unique_char)

unique_char.clear()
print(unique_char)
print(char_list.discard('d'))
print(char_list.remove('d'))

print(char_list.difference({'a', 'e', 'i'}))
print(char_list.intersection({'a', 'e', 'i'}))