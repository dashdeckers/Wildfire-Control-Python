from tkinter import Tk
import gui
import state
import agent

app = Tk()

env = state.Environment(50, 50)

grid = gui.Grid(app, env)
grid.pack()

app.mainloop()