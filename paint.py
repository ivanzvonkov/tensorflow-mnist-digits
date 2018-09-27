import numpy as np
from Tkinter import *

class Paint:

    def paint(self, event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.cv.create_rectangle(x1, y1, x2, y2, fill='black', width=5)
        self.update_drawing_feature(event.x//4, event.y//4)


    def clear(self):
        self.cv.delete('all')

    def guess(self):
        self.observer(self.drawing_feature)

    def update_drawing_feature(self,x,y):
        self.drawing_feature[x][y] = 1

    def __init__(self, observer):
        self.width = 28 * 4
        self.height = 28 * 4
        self.center = self.height // 2
        self.white = (255, 255, 255)
        self.drawing_feature = np.zeros((28, 28))
        self.observer_fn = observer

        root = Tk()

        # Tkinter Create canvas
        self.cv = Canvas(root, width=self.width, height=self.height, bg='white')
        self.cv.pack

        self.cv.pack(expand=YES, fill=BOTH)
        self.cv.bind('<B1-Motion>', self.paint)

        clear_button = Button(text='Clear', command=self.clear)
        guess_button = Button(text='Guess', command=self.guess)

        clear_button.pack()
        guess_button.pack()

        root.mainloop()

