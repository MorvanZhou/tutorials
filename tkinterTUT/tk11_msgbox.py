# View more python learning tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

import tkinter
import tkinter.messagebox

window = tkinter.Tk()
window.title('my window')
window.geometry('200x200')

def hit_me():
    #tk.messagebox.showinfo(title='Hi', message='hahahaha')   # return 'ok'
    #tk.messagebox.showwarning(title='Hi', message='nononono')   # return 'ok'
    #tk.messagebox.showerror(title='Hi', message='No!! never')   # return 'ok'
    #print(tk.messagebox.askquestion(title='Hi', message='hahahaha'))   # return 'yes' , 'no'
    #print(tk.messagebox.askyesno(title='Hi', message='hahahaha'))   # return True, False
    print(tkinter.messagebox.asktrycancel(title='Hi', message='hahahaha'))   # return True, False
    print(tkinter.messagebox.askokcancel(title='Hi', message='hahahaha'))   # return True, False
    print(tkinter.messagebox.askyesnocancel(title="Hi", message="haha"))     # return, True, False, None

tkinter.Button(window, text='hit me', command=hit_me).pack()
window.mainloop()









