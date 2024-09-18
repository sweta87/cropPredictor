from tkinter import *
import os
# from gui_stuff import *


import pandas as pd
import numpy as np
from PIL import ImageTk,Image


def run_cp():
    root.destroy()
    import crop_predictor.py

def run_fp():
    root.destroy()
    import fertilizer_predictor.py
    

root = Tk()
#root.geometry("400*200")
root.title("KRISHI")
#root.configure(background='white')

img = Image.open("images/swobalilogo.png")
logo = img.resize((80, 100), Image.LANCZOS)
logo = ImageTk.PhotoImage(logo)
label = Label(root, image = logo)
label.image = logo
label.grid(row=1, column=0, columnspan=4,pady=20, padx=50)


head = Label(root, justify=LEFT, text="KRISHI", fg="Dark green" )
head.config(font=("Elephant", 32,))
head.grid(row=1, column=0, columnspan=4, padx=100)

head2 = Label(root, justify=LEFT, text="Crop & Fertilizer Predictor", fg="black" )
head2.config(font=("Elephant", 20,"bold"))
head2.grid(row=2, column=0, columnspan=4, padx=100)

btn_cp = Button(root, text="Predict Crop", command=run_cp,bg="White",fg="Dark red", pady=30, width=25)
btn_cp.config(font=("Times new roman", 22))
btn_cp.grid(row=3, column=2,padx=50,pady=20)

btn_fp = Button(root, text="Predict Fertilizer", command=run_fp,bg="White",fg="Dark red", pady=30, width=25)
btn_fp.config(font=("Times new roman", 22))
btn_fp.grid(row=4, column=2,padx=50,pady=20)


root.mainloop()
