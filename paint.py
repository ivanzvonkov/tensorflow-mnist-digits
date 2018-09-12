import PIL
from PIL import Image, ImageTk, ImageDraw
from Tkinter import *

width = 28*5
height = 28*5
center = height//2
white = (255,255,255)


def paint(event):
    x1, y1 = (event.x -1), (event.y -1)
    x2,y2 = (event.x +1), (event.y +1)
    cv.create_oval(x1,y1,x2,y2,fill='black', width=5)
    draw.line([x1, y1, x2, y2], fill='black', width=5)

def clear():
    drawing_image = PIL.Image.new('RGB', (width, height), white)


root = Tk()

# Tkinter Create canvas
cv = Canvas(root, width=width,height=height, bg='white')
cv.pack

# PIL create empty image and draw object to draw on
drawing_image = PIL.Image.new('RGB', (width, height), white)
draw = ImageDraw.Draw(drawing_image)

#cv.create_line([0,center, width, center], fill='black')

cv.pack(expand=YES, fill=BOTH)
cv.bind('<B1-Motion>', paint)

button = Button(text='Clear', command=clear)

button.pack()

root.mainloop()