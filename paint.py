import PIL
from PIL import Image, ImageTk, ImageDraw
from Tkinter import *

width = 28 * 5
height = 28 * 5
center = height // 2
white = (255, 255, 255)

# PIL create empty image and draw object to draw on
#drawing_image = PIL.Image.new('RGB', (width, height), white)
#draw = ImageDraw.Draw(drawing_image)


def paint(event):
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    cv.create_rectangle(x1, y1, x2, y2, fill='black', width=5)


def clear():
    cv.delete('all')



def guess():
    print 'guessing'



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
