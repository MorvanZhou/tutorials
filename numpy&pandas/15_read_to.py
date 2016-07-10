# View more python tutorials on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

import pandas as pd

# read from
data = pd.read_csv('student.csv')
print(data)

# save to
data.to_pickle('student.pickle')