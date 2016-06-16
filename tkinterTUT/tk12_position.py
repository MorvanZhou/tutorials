# View more python learning tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

import tkinter as tk

window = tk.Tk()
window.geometry('200x200')

#canvas = tk.Canvas(window, height=150, width=500)
#canvas.grid(row=1, column=1)
#image_file = tk.PhotoImage(file='welcome.gif')
#image = canvas.create_image(0, 0, anchor='nw', image=image_file)

#tk.Label(window, text='1').pack(side='top')
#tk.Label(window, text='1').pack(side='bottom')
#tk.Label(window, text='1').pack(side='left')
#tk.Label(window, text='1').pack(side='right')

#for i in range(4):
    #for j in range(3):
        #tk.Label(window, text=1).grid(row=i, column=j, padx=10, pady=10)

tk.Label(window, text=1).place(x=20, y=10, anchor='nw')

window.mainloop()
