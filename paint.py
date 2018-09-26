import numpy as np
from Tkinter import *

width = 28 * 4
height = 28 * 4
center = height // 2
white = (255, 255, 255)

drawing_feature = np.zeros((28, 28))

def paint(event):
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    cv.create_rectangle(x1, y1, x2, y2, fill='black', width=5)
    update_drawing_feature(event.x//4, event.y//4)


def clear():
    cv.delete('all')

def guess():
    print 'guessing'
    print drawing_feature

def update_drawing_feature(x,y):
    drawing_feature[x][y] = 1

root = Tk()

# Tkinter Create canvas
cv = Canvas(root, width=width, height=height, bg='white')
cv.pack

cv.pack(expand=YES, fill=BOTH)
cv.bind('<B1-Motion>', paint)

clear_button = Button(text='Clear', command=clear)
guess_button = Button(text='Guess', command=guess)

clear_button.pack()
guess_button.pack()

root.mainloop()
