'''
Animation of the density of a Gaussian Copula (Bi-variate case).
The animation changes the Rho (correlation) between the variables, and rotates the figure at the same time.
by: Mauricio Salazar
Date: 04-Nov-2019
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import axes3d, Axes3D
from scipy.stats import norm, gamma, pareto, multivariate_normal
import seaborn as sns


def ranks(n=2000, rho=0.2, spear=True):
    """
    Function to create vectors of rank with known rank correlation (Spearman's rho)
        --- Input
            n: Number of observations desired, default is 2000
            rho: Correlation coefficient. If Spear==False then this parameter enters directly into the covariance matrix, R.
            spear: If True then rho is rescaled to fit the Gaussian copula.
        --- Output:
            Two uniformly distributed vectors with desired Spearman's rho.
    """

    if spear:
        rho_G = 2 * np.sin(rho * np.pi / 6)
    else:
        rho_G = rho

    R = np.array([[1, rho_G],
                  [rho_G, 1]])

    C = norm.cdf(np.random.multivariate_normal([0, 0], R, size=n))
    U, V = C[:, 0], C[:, 1]

    return U, V

# Example of a Copula with fixed Rho
marker_size = 7
rho = 0.8

U,V = ranks(n=5000,rho=rho)
sns.jointplot(U,V,s=marker_size).set_axis_labels("U", "V");

distX = pareto(b=5)
distY = gamma(10, loc = 0., scale = 2.)
plt.show()

X = distX.ppf(U)
Y = distY.ppf(V)

sns.jointplot(X,Y,s=marker_size).set_axis_labels("X", "Y")
plt.show()


# Density of Gaussian Copula
resolution = 1/500

U, V = np.mgrid[0.05:0.95:resolution, 0.05:0.95:resolution]
pos = np.dstack((U, V))

bi_gaussian = multivariate_normal([0.0, 0.0], [[1, rho], [rho, 1]])
gaussian = norm(loc=0, scale=1)
c_density = bi_gaussian.pdf(gaussian.ppf(pos)) / (gaussian.pdf(gaussian.ppf(U)) * gaussian.pdf(gaussian.ppf(V)))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(U, V, c_density, edgecolors='k', linewidth=0.2, cmap='Blues')
ax.set_xlim([0,1])
ax.set_ylim([0,1])
ax.set_zlim([0,4])
ax.set_xlabel('U')
ax.set_ylabel('V')
ax.set_zlabel('c(u,v)')
ax.set_title('Conditional copula density - c(u,v)')
ax.view_init(20, 100)
plt.show()

#%% Animation
frames = 360
rho_vector = np.linspace(0.3,0.8,frames)
zarray = np.zeros((U.shape[0], V.shape[0], frames))

for ii, rho in enumerate(rho_vector):
    bi_gaussian = multivariate_normal([0.0, 0.0], [[1, rho], [rho, 1]])
    zarray[:,:,ii] = bi_gaussian.pdf(gaussian.ppf(pos)) / (gaussian.pdf(gaussian.ppf(U)) * gaussian.pdf(gaussian.ppf(V)))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plot = [ax.plot_surface(U, V, zarray[:,:,0], edgecolors='k', linewidth=0.2, cmap='Blues')]
ax.set_xlim([0,1])
ax.set_ylim([0,1])
ax.set_zlim([0,4])
ax.set_xlabel('U')
ax.set_ylabel('V')
ax.set_zlabel('c(u,v)')
#ax.set_title('Conditional copula density - c(u,v)')
ax.view_init(20, 100)
#plt.show()

def update_plot(frame_number, zarray, rho_vector, plot):
    plot[0].remove()
    plot[0] = ax.plot_surface(U, V, zarray[:,:,frame_number], edgecolors='k', linewidth=0.2, cmap='Blues')
    ax.view_init(elev=20., azim=100+frame_number/2.)
    ax.set_title('Conditional copula density - c(u,v) - Rho: {:.2f}'.format(rho_vector[frame_number]))
    # ax.text2D(0.8, 0.9, s='Rho: {:.2f}'.format(rho_vector[frame_number]), transform=ax.transAxes)

animate = animation.FuncAnimation(fig, update_plot, fargs=(zarray, rho_vector, plot),frames=360, interval=20)
# FFMpegWriter = animation.writers['ffmpeg']
# writervideo = FFMpegWriter(fps=30, extra_args=['-vcodec', 'libx264'])
#
# writervideo = animation.FFMpegWriter(fps=30)
animate.save(r'animations\basic_animation6.mp4', fps=30, writer="ffmpeg")
# animate.save(r'animations\basic_animation6.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
# animate.save('basic_animation6.mp4', fps=30)
# animate.save(r'animations\animation.gif', writer='PillowWriter', fps=30)
# animate.save('basic_animation50.mp4',bitrate=500, fps=30, extra_args=['-vcodec', 'libx264'])
plt.show()