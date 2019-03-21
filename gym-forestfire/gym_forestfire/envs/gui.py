from tkinter import *


class Cell():
    def __init__(self, master, x, y, size, element):
        self.master = master
        self.x = x
        self.y = y
        self.size = size
        self.element = element

    def draw(self, draw_agent=False):
        if self.master is not None:
            if draw_agent:
                fill = "Pink"
            else:
                fill = self.element.get_color()
            outline = "Black"

            xmin = self.x * self.size
            xmax = xmin + self.size
            ymin = self.y * self.size
            ymax = ymin + self.size

            self.master.create_rectangle(xmin, ymin, xmax, ymax, fill=fill, outline=outline)


class Grid(Canvas):
    def __init__(self, master, env):
        self.cell_size = 10
        self.rows = env.height
        self.cols = env.width
        self.env = env

        width = self.cell_size * self.cols
        height = self.cell_size * self.rows

        Canvas.__init__(self, master, width=width, height=height)

        self.grid = []
        for row in range(self.rows):
            line = []
            for col in range(self.cols):
                element = self.env.get_at(row, col)
                line.append(Cell(self, col, row, self.cell_size, element))
            self.grid.append(line)

        self.bind("<Button-1>", self.print_cell)  
        self.bind_all("<space>", self.update_simulation)
        self.draw()

    def draw(self):
        agent_coords = self.env.get_agent_coords()
        for row in range(len(self.grid)):
            for col in range(len(self.grid[0])):
                if (row, col) in agent_coords:
                    self.grid[row][col].draw(draw_agent=True)
                else:
                    self.grid[row][col].draw()

    def _eventCoords(self, event):
        row = int(event.y / self.cell_size)
        column = int(event.x / self.cell_size)
        return row, column

    def print_cell(self, event):
        row, col = self._eventCoords(event)
        cell = self.grid[row][col].element
        print(cell.type)

    def update_simulation(self, event):
        if self.env.running:
            self.after(10, self.env.update())
            self.draw()
