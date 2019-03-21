from tkinter import Tk
import gui
import state
import sys

if len(sys.argv) > 0 and sys.argv[1] == "no_gui":
    import timeit
    print("Running headless")

    setup = '''
    import state
    env = state.Environment(50,50)
    '''
    code = '''
    env.reset_world()
    while env.running:
        env.update()
    '''
    num_runs = 100

    total = timeit.timeit(setup=setup, stmt=code, number=num_runs)
    print("Total:", total, "Average per run", total / num_runs)

else:
    print("Running with GUI")
    app = Tk()
    env = state.Environment(50, 50)

    grid = gui.Grid(app, env)
    grid.pack()

    app.mainloop()
