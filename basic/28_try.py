# View more python learning tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

try:
    file = open('eeee', 'r+')
except Exception as e:
    print('there is no file named as eeeee')
    response = input('do you want to create a new file')
    if response =='y':
        file = open('eeee','w')
    else:
        pass
else:
    file.write('ssss')
file.close()


