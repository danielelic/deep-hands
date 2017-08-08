from tkinter import *

import PIL.Image
import PIL.ImageTk


class Map(Canvas):
    def __init__(self, master, scale=1, **kw):
        Canvas.__init__(self, master, **kw)
        self.scale = scale

        # Rightside Log
        self.T = Text(self, height=8, width=55)
        self.T.place(relx=0.5, rely=0.97, anchor=S)
        self.T.config(state="disabled")

    def draw_image(self, path_image):
        im = PIL.Image.open(path_image)
        photo = PIL.ImageTk.PhotoImage(im)
        self.photo = photo
        self.master.img = PhotoImage(path_image)
        self.create_image(self.master.winfo_width() / 2, self.master.winfo_height() / 3, anchor=CENTER, image=photo)

    def log(self, txt):
        self.T.config(state="normal")
        self.T.insert(END, txt)
        self.T.config(state="disabled")

    def clear_log(self):
        self.T.config(state="normal")
        self.T.delete(1.0, END)
        self.T.config(state="disabled")
