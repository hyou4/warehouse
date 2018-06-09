
from Tkinter import *

class Application(Frame):
    def __init__(self):
        super(self).__init__()
        self.initUI()

    def initUI(self):
        self.master.title("EECS221 App")
        self.pack(fill=BOTH, expand=1)

