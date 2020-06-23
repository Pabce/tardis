import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection

import tools

class Plotter:

    def __init__(self, figure, axis, A, R, ALPHA):
        self.fig = figure
        self.ax = axis
        self.A = A
        self.R = R
        self.ALPHA = ALPHA

        self.has_colorbar = False
        self.has_bubble = False
        self.has_title = False
        self.has_legend = False


    def plot_bubble(self):
        A, R = self.A, self.R
        ax = self.ax

        ext_c = plt.Circle((0, 0), (A ** 2 + R ** 2) ** 0.5, fill=False) #, linewidth=3)
        int_c = plt.Circle((0, 0), (A ** 2 - R ** 2) ** 0.5, fill=False) #, linewidth=3)

        ax.add_artist(ext_c)
        ax.add_artist(int_c)

        ax.set_xlim([-150, 150])
        ax.set_ylim([-150, 150])
        ax.set_aspect('equal')

        # Small: 20, 22
        # Medium:
        ax.tick_params(axis='both', which='major', labelsize=16)

        self.ax.set_xlabel(r'$x$', fontsize=18)
        self.ax.set_ylabel(r'$t$', rotation=0, fontsize=18)

        plt.grid(animated=True)
        plt.tight_layout()

        self.has_bubble = True


    # Weird code to achieve a multicoloured continuous line. Amazingly it works...
    @staticmethod
    def get_solid_line(y, c_map, width, color, onecolor=None, limits=None):
        points = np.array([y[3, :], y[2, :]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        if onecolor:
            color = np.array([onecolor] * len(y[0, :]))
            lc = LineCollection(segments, colors=color)
        else:
            if limits:
                norm = plt.Normalize(limits[0], limits[1])
                lc = LineCollection(segments, cmap=c_map, norm=norm)
            else:
                lc = LineCollection(segments, cmap=c_map)

            lc.set_array(color)

        lc.set_linewidth(width)

        #lc.set_joinstyle('bevel')
        lc.set_snap(False)
        lc.set_capstyle('round') #projecting
        lc.set_antialiased(True)

        return lc

    # Specify either a color (onecolor) or a colormap
    def plot_trajectory(self, y, lam, step, error, limits=None,
                        colormap=None, onecolor=None, colorbar="plot", solid=False, width=2, nature="geodesic"):
        # Get type of trajectory:
        m = tools.get_squared_mod(*y[:,0])
        mod = 0
        if m < -0.01:
            mod = -1

        if nature == "ctc":
            y = tools.cut_outside_bubble(y)


        # Get the colormap!
        color = []
        cbtitle = ''
        c_map = 'gnuplot' #nipy_spectral_r #jet_r #gnuplot #plasma

        if colormap == "energy": # E/Rest mass (or simply E/E0 = f/f0 for m=0 particles)
            color = tools.get_energy_on_trajectory(y)
            cbtitle = r'$E/m$'
            if mod == 0:
                cbtitle = r'$E/E_0 = f/f_0$'
                color /= color[0]
        elif colormap == "log_energy":
            lcolor = tools.get_energy_on_trajectory(y)
            color = np.log10(lcolor)
            cbtitle = r'$\log {(E/m)}$'
            if mod == 0:
                cbtitle = r'$\log{(E/E_0)} = \log{(f/f_0)}$'
                lcolor /= lcolor[0]
                color = np.log10(lcolor)
        elif colormap == "k_energy": # T/Rest mass (Only for massive particles)
            cbtitle = r'$T/m$'
            if mod == 0:
                print("Beware: This colormap is not well defined for massless particles!")
            color = tools.get_energy_on_trajectory(y) - 1
        elif colormap == "local_speed": # speed as seen by an observer in a local inertiar reference frame at rest w/ respect to the original coordinate system (with respect to the bubble)
            cbtitle = r'$v$ (local)'
            color = tools.get_3speed_on_trajectory(y, local=True)
        elif colormap == "local_speed_angle":
            cbtitle = r'$v$ angle (º) (local)'
            color = tools.get_3speed_on_trajectory(y, local=True, angle=True)
        elif colormap == "ext_speed":
            cbtitle = r'$v$ (º) (external obsever)'
            color = tools.get_3speed_on_trajectory(y, local=False)
        elif colormap == "ext_speed_angle":
            cbtitle = r'$v$ angle (º) (external observer)'
            color = tools.get_3speed_on_trajectory(y, local=False, angle=True)
        elif colormap == "ext_speed_supra":
            color = tools.get_3speed_on_trajectory(y, local=False, angle=False)
            for i in range(len(color)):
                color[i] = 0 if color[i] < 1-10e-5 else 1
            c_map = 'brg_r'
        elif colormap == "mod":
            color = tools.get_squared_mod(*y)
        elif colormap == "step_size":
            cbtitle = r'Step size'
            color = step

        elif type(colormap) == list:
            start, end = tools.get_origin(y)
            colors = ['red', 'green', 'blue', '#ff8c00', 'purple']

            # Set legend
            if not self.has_legend:
                patches = []
                all_patches = [mpatches.Patch(color=colors[0], label='Starts and ends at past infinity'),
                mpatches.Patch(color=colors[1], label='Starts and ends at future infinity'),
                mpatches.Patch(color=colors[2], label='From past to future infinity'),
                mpatches.Patch(color=colors[3], label='From past infinity to singularity'),
                mpatches.Patch(color=colors[4], label='From future infinity to singularity')]
                for i in range(len(colors)):
                    if colormap[i] == 1:
                        patches.append(all_patches[i])
                self.ax.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.5, -0.07), ncol=3)

                self.has_legend = True
                self.has_colorbar = True

            # Start and end at same endpoints
            if start == end:
                if start == -1:
                    if colormap[0] == 0:
                        return
                    onecolor = colors[0]
                else:
                    if colormap[1] == 0:
                        return
                    onecolor = colors[1]
            # 'Normal' curves
            elif start + end == 0:
                if colormap[2] == 0:
                    return
                onecolor = colors[2]
            # Start or end at singularities
            elif end == 0:
                if start == - 1:
                    if colormap[3] == 0:
                        return
                    onecolor = colors[3]

                elif start == 1:
                    if colormap[4] == 0:
                        return
                    onecolor = colors[4]
            else:
                onecolor = "black"

        # Plot the line
        if solid:
            lc = self.get_solid_line(y, c_map, width, color, onecolor=onecolor,limits=limits)
            img = self.ax.add_collection(lc)
        else:
            if onecolor:
                print("Onecolor not supported yet for scatter plot. Sorry!")
                exit()
            if limits:
                img = self.ax.scatter(y[3, :], y[2, :], s=width, c=color, cmap=c_map, vmin=limits[0], vmax=limits[1])
            else:
                img = self.ax.scatter(y[3, :], y[2, :], s=width, c=color, cmap=c_map)


        # Plot the colorbar
        if colorbar == "plot":
            c = plt.colorbar(img)
            c.ax.set_title(cbtitle)
        if colorbar == "once" and not self.has_colorbar:
            if np.max(color) > limits[1]*1.001:
                print(np.max(color))
                c = plt.colorbar(img) # orientation="horizontal", fraction=0.046, pad=0.16)#, extend='both')
                c.ax.tick_params(labelsize=14)
            else:
                #c = plt.colorbar(img)
                c = plt.colorbar(img) #, orientation="horizontal", fraction=0.046, pad=0.12)
                c.ax.tick_params(labelsize=14)

            c.ax.set_title(cbtitle, fontsize='15')
            self.has_colorbar = True
            if not limits:
                print("Warning! You have set no limits, hence the colorbar is only representative of one trajectory!")

        # Set title
        if not self.has_title:
            geo = 'Null' if mod == 0 else 'Timelike'
            nat = 'geodesic' if nature == 'geodesic' else 'curve'
            sing = 's' if colorbar == 'once' else ''
            title = '{} {}{}'.format(geo, nat, sing)

            self.ax.set_title(title, fontsize=15)
            self.has_title = True


    def get_titles(self, colormap, c_map, mod):

        return


    def plot_secondary(self, y, lam, step, error):
        return


    # Plot tangent vector at a given point
    def plot_tangent_vector(self, trajectory, t, x):
        # y = trajectory
        #
        # self.ax.arrow(x, t, d[j][1] * size, d[j][0] * size, head_width=1.5, width=0.15, color="orange")
        return


    def plot_lightcone(self, t, x, size):
        d = tools.find_local_lightcone(t, x)

        for j in range(len(d)):
            self.ax.arrow(x, t, d[j][1] * size, d[j][0] * size, head_width=1.5*size/6, width=0.1*size/6, color="orange")


    def plot_time_arrow(self, t, x, size):
        v = tools.find_local_rest_velocity(t, x)

        norm = (v[1][0]**2+v[1][1]**2)**0.5

        for j in range(1, len(v)):
            self.ax.arrow(x, t, v[j][1] * size/norm, v[j][0] * size/norm, head_width=1.5*size/6, width=0.1*size/6, color="blue")


    # Plot a series of lightcones, time arrows or tangent vectors
    # Specify either a radius (not for tangent vectors) or a trajectory and a parameter on it (lam).
    def plot_light_time(self, num, size, lightcones=False, time_arrows=False, tangents=False, param_direction=False,
                        radius=None, trajectory=None, param=None):
        fig, ax = self.fig, self.ax

        if radius:
            for i in range(num):
                theta = 2 * np.pi * i / num
                x = radius * np.cos(theta)
                t = radius * np.sin(theta)

                if lightcones:
                    self.plot_lightcone(t, x, size)
                if time_arrows:
                    self.plot_time_arrow(t,x, size)

        if trajectory is not None and param is not None:
            y = trajectory

            # Idea: plot arrows in an equispaced way, only in the visible space.
            length = len(y[0, :])
            cutoff = length - 1
            for i in range(length):
                if abs(y[2, i]) > 150 or abs(y[3, i]) > 150:
                    cutoff = i
                    break

            spacing = (param[cutoff] - param[0]) / num

            n = 0
            for i in range(cutoff):
                if param[i] > spacing * n:
                    n+=1
                else:
                    continue

                z = y[: , i]
                u0, u1, t, x = z[0], z[1], z[2], z[3]

                if lightcones:
                    self.plot_lightcone(t, x, size)
                if time_arrows:
                    self.plot_time_arrow(t, x, size)
                if param_direction:
                    norm = (u0**2 + u1**2)**0.5
                    u0 /= norm
                    u1 /= norm
                    self.ax.arrow(x, t, u1, u0, head_width=5, head_length=3, width=0.15, color="black", shape="full", length_includes_head=False, zorder=2)
                    #self.ax.annotate(r'$\tau$', xy=(x+2, t+2))
                    #ax.annotate(r'$\tau$', xy=(x+u1, t+u0), xytext=(x, t), arrowprops={'arrowstyle': '->', "color":'red'})
                if tangents:
                    norm = (u0 ** 2 + u1 ** 2) ** 0.5
                    u0 /= norm
                    u1 /= norm
                    self.ax.arrow(x, t, u1*size, u0*size, head_width=2.5, width=0.3, color="black", shape="full", zorder=2)
