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


