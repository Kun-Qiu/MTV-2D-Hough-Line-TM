import numpy as np
import matplotlib.pyplot as plt

class GridInspector:
    def __init__(self, ref, mov, grid_T0, grid_dT):
        self.ref = ref
        self.mov = mov
        self.grid_T0 = grid_T0
        self.grid_dT = grid_dT

        self.active_point = None
        self.active_ax = None

    def launch(self, i_sel, j_sel):
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 5))
        ax0.imshow(self.ref, cmap="gray")
        ax1.imshow(self.mov, cmap="gray")

        ax0.set_title("Reference")
        ax1.set_title("Moving")

        self.points_T0 = []
        self.points_dT = []

        # --- plot all points ---
        for i in range(self.grid_T0.grid.shape[0]):
            for j in range(self.grid_T0.grid.shape[1]):
                if self.grid_T0.grid[i, j] is None:
                    continue

                x0, y0 = self.grid_T0.grid[i, j]
                x1, y1 = self.grid_dT.grid[i, j]

                p0 = ax0.plot(x0, y0, "ro", alpha=0.3, picker=5)[0]
                p1 = ax1.plot(x1, y1, "bo", alpha=0.3, picker=5)[0]

                p0.meta = (i, j, "T0")
                p1.meta = (i, j, "dT")

                self.points_T0.append(p0)
                self.points_dT.append(p1)

        # --- highlight selected ---
        x0, y0 = self.grid_T0.grid[i_sel, j_sel]
        x1, y1 = self.grid_dT.grid[i_sel, j_sel]

        self.hl_T0 = ax0.plot(x0, y0, "yo", markersize=10)[0]
        self.hl_dT = ax1.plot(x1, y1, "yo", markersize=10)[0]

        fig.canvas.mpl_connect("pick_event", self.on_pick)
        fig.canvas.mpl_connect("motion_notify_event", self.on_drag)
        fig.canvas.mpl_connect("button_release_event", self.on_release)

        plt.show()

    def on_pick(self, event):
        self.active_point = event.artist
        self.active_ax = event.mouseevent.inaxes

        i, j, which = self.active_point.meta

        x0, y0 = self.grid_T0.grid[i, j]
        x1, y1 = self.grid_dT.grid[i, j]

        self.hl_T0.set_data(x0, y0)
        self.hl_dT.set_data(x1, y1)

    def on_drag(self, event):
        if self.active_point is None or event.inaxes != self.active_ax:
            return

        i, j, which = self.active_point.meta
        x, y = event.xdata, event.ydata

        if x is None or y is None:
            return

        self.active_point.set_data(x, y)

        if which == "T0":
            self.grid_T0.grid[i, j] = (x, y)
            self.hl_T0.set_data(x, y)
        else:
            self.grid_dT.grid[i, j] = (x, y)
            self.hl_dT.set_data(x, y)

        self.active_point.figure.canvas.draw_idle()

    def on_release(self, event):
        self.active_point = None
        self.active_ax = None
