import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib

possibilities = ["both_above","both_below","opposite_above","opposite_below","opposite_together","both_together"]
norm_correction = 1 # this needs fixed, trying to make it so that it make the weak asymptotes still very visual, but not make the zero points visible. needs applied to U and V


class GIQLFlow():
    d = 23
    x_min, y_min = -0.1,-0.1
    x_max, y_max = 5,5

    def __init__(self,i):
        self.i = i
        x = np.linspace(self.x_min, self.x_max, self.d)
        y = np.linspace(self.y_min, self.y_max, self.d)
        self.X, self.Y = np.meshgrid(x, y)
        self.xlim=[self.x_min, self.x_max]
        self.ylim=[self.y_min, self.y_max]
        inferno = plt.cm.inferno(np.linspace(0,1,100))
        white = plt.cm.seismic(np.ones(1)*0.5)
        colors = np.vstack((white,inferno))
        self.my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('my_map', colors)
        self.U,self.V = self.which_field(i,0)#init


    def sigmoid(self,x):
        return 1.0 / (1 + np.exp(-x))
    def smooth(self,t, inflection=10):
        error = self.sigmoid(-inflection / 2)
        return np.clip((self.sigmoid(inflection * (t - 0.5)) - error) / (1 - 2 * error),0, 1)
    def norm2(self,U,V):
        return np.sqrt(U**2+V**2)

    def which_field(self,i,eps):
        eps_0 = eps/2
        eps_1 = (1-eps/2)
        if i == "both_above":
            output = eps_0*(eps_0+4*eps_1 - self.X), eps_1*(3*eps_1 - self.Y)
            output = np.where(self.Y-self.X > 0,output,0)
        if i == "both_below":
            output = eps_1*(eps_1+4*eps_0 - self.X), eps_0*(3*eps_0 - self.Y)
            output = np.where(self.Y-self.X < 0,output,0)
        if i == "opposite_below":
            output = eps_1*(eps_0+4*eps_1 - self.X), eps_0*(3*eps_1 - self.Y)
            output = np.where(self.Y-self.X < 0,output,0)
        if i == "opposite_above":
            output = eps_0*(eps_1+4*eps_0 - self.X), eps_1*(3*eps_0 - self.Y)
            output = np.where(self.Y-self.X > 0,output,0)
        if i == "opposite_together":
            output1 = eps_1*(eps_0+4*eps_1 - self.X), eps_0*(3*eps_1 - self.Y)
            output1 = np.where(self.Y-self.X < 0,output1,0)
            output2 = eps_0*(eps_1+4*eps_0 - self.X), eps_1*(3*eps_0 - self.Y)
            output2 = np.where(self.Y-self.X > 0,output2,0)
            output = output1 + output2
        if i == "both_together":
            output1 = eps_0*(eps_0+4*eps_1 - self.X), eps_1*(3*eps_1 - self.Y)
            output1 = np.where(self.Y-self.X > 0,output1,0)
            output2 = eps_1*(eps_1+4*eps_0 - self.X), eps_0*(3*eps_0 - self.Y)
            output2 = np.where(self.Y-self.X < 0,output2,0)
            output = output1 + output2
        return output

    def update_quiver(self,eps):
        U, V = self.which_field(self.i,self.smooth(eps))
        colors = np.sqrt(U**2+V**2)
        self.Q.set_UVC(U,V, self.norm2(U,V))#,colormap(norm(colors)))
        return self.Q

    def create_anim(self):
        fig, ax = plt.subplots(1,1)
        ax = plt.axes(xlim=self.xlim,ylim=self.ylim)
        ax.set_title("%s"%i)
        self.Q = ax.quiver(self.X, self.Y, self.U*norm_correction, self.V*norm_correction, self.norm2(self.U,self.V),cmap=self.my_cmap,pivot='mid')
        anim = animation.FuncAnimation(fig, self.update_quiver, frames = np.linspace(1,0,100),interval = 70, blit=False,repeat = True) #eps reversed here
        fig.tight_layout()
        plt.show()

for i in possibilities:
    GIQLFlow(i).create_anim()
ani.save("pdflow.gif",writer="imagemagick")
