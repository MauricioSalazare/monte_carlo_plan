from scipy.optimize import leastsq
import numpy as np
import plotly.graph_objects as go
from typing import Tuple
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import vonmises, norm
from math import fmod

def fit_sphere(coords_pca):
    """Compute the center and radius of sphere based on a dataset of points"""

    def fitfunc(p, coords):
        x0, y0, z0, _ = p
        x, y, z = coords.T
        return np.sqrt((x-x0)**2 + (y-y0)**2 + (z-z0)**2)

    # p0 = [x0, y0, z0, R]
    p0 = [0, 0, 0, 1]
    errfunc = lambda p, x: fitfunc(p, x) - p[3]
    p1, flag = leastsq(errfunc, p0, args=(coords_pca,))

    return p1[:3], p1[3]

def create_sphere_surface(center, radius):
    """Creates surface coordinates of a sphere shifted by center coordinates"""
    phi, theta = np.mgrid[0.0:np.pi:100j, 0.0:2.0 * np.pi:100j]
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)

    x += center[0]
    y += center[1]
    z += center[2]

    return x, y, z

def polar_coordinates(coords_pca, ax=None):
    '''
    Polar coordinates are referenced as ISO.
        R: radius from the center
        theta: Angle from the z+ axis
        phi: Azimuth from the x+ axis
    '''

    r_pca = np.sqrt(coords_pca[:, 0] ** 2 + coords_pca[:, 1] ** 2 + coords_pca[:, 2] ** 2)

    # -------------------------------------------------------------------------------------
    ## PHI CONVERSION (AZIMUTHAL ANGLE)
    phi_pca = np.arctan(coords_pca[:, 1] / coords_pca[:, 0])
    phi_pca_degrees = np.degrees(phi_pca)

    # Rules of quadrants for azimuthal angle.
    idx_quadrant_3_4 = coords_pca[:, 0] < 0.0
    idx_quadrant_2 = (coords_pca[:, 0] > 0.0) & (coords_pca[:, 1] < 0.0)

    # Change the reference (The positive X axis is the reference point)
    phi_pca_degrees[idx_quadrant_2] += 360.0
    phi_pca_degrees[idx_quadrant_3_4] += 180.0

    # -------------------------------------------------------------------------------------
    ## THETA CONVERSION (POLAR ANGLE)
    theta_pca = np.arctan(np.sqrt((coords_pca[:, 0] ** 2 + coords_pca[:, 1] ** 2)) / coords_pca[:, 2])
    theta_pca_degrees = np.degrees(theta_pca)

    # Rules for the polar angle
    idx_quadrant_5_6_7_8 = coords_pca[:, 2] < 0.0

    # Change the reference (The positive Z axis is the reference point)
    theta_pca_degrees[idx_quadrant_5_6_7_8] += 180
    if ax is None:
        fig, ax = plt.subplots(1, 4, figsize=(16, 4))
        plt.subplots_adjust(bottom=0.15, wspace=0.3, left=0.1, right=0.9)


    sns.histplot(data=r_pca, stat="density", kde=True, bins="fd", element="step", ax=ax[0])
    sns.rugplot(x=r_pca, ax=ax[0])
    sns.histplot(data=phi_pca_degrees, stat="density", kde=True, bins="fd", element="step", ax=ax[1])
    sns.rugplot(x=phi_pca_degrees, ax=ax[1])
    sns.histplot(data=theta_pca_degrees, stat="density", kde=True, bins="fd", element="step", ax=ax[2])
    sns.rugplot(x=theta_pca_degrees, ax=ax[2])

    ax[0].set_title('Radius')
    ax[0].set_xlabel('Radius magnitude')
    ax[0].set_ylabel('Normalized density [-]')
    ax[0].tick_params(axis='y', which='both', left=False, top=False, labelleft=False)

    ax[1].set_title('Phi - Azimuthal angle')
    ax[1].set_xlabel('Degrees [°]')
    ax[1].set_xlim([0, 360])
    # ax[1].set_ylabel('Normalized density [-]')
    ax[1].tick_params(axis='y', which='both', left=False, top=False, labelleft=False)

    ax[2].set_title('Theta - Polar angle')
    ax[2].set_xlabel('Degrees [°]')
    ax[2].set_xlim([0, 180])
    # ax[2].set_ylabel('Normalized density [-]')
    ax[2].tick_params(axis='y', which='both', left=False, top=False, labelleft=False)

    # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    sns.scatterplot(x=phi_pca_degrees, y=theta_pca_degrees, ax=ax[3])
    ax[3].set_xlim([0, 360])
    ax[3].set_ylim([0, 180])
    ax[3].set_xlabel('Phi - Azimuth')
    ax[3].set_ylabel('Theta - Polar')

    return (r_pca, phi_pca_degrees, theta_pca_degrees)

def reject_outliers(data, m = 2):
    """Reject data that are "m" times outside the median

    The returned indices are the GOOD data points

    Returns:
    --------
        Tuple[np.ndarray, np.ndarray] = Boolean indices of the data points to reject

    """

    ad = np.abs(data - np.median(data))  # Absolute deviations from the median
    mad = np.median(ad)  # Median absolute deviation (MAD)
    mad = 1.4826 * np.median(ad)
    s = ad/mad if mad else 0.
    return (s < m, data[s < m])

def reject_outliers_mean(data: np.ndarray, m: int =2) -> Tuple[bool, np.ndarray]:
    """
    Find indices of data that are "m" times standard deviations inside the median.
    The returned indices are the GOOD data points

    Returns:
    --------
        Tuple[np.ndarray, np.ndarray] = Boolean indices of the data points to reject

    """

    return (abs(data - np.mean(data)) < m * np.std(data),
           data[abs(data - np.mean(data)) < m * np.std(data)])

def fit_angle_distributions(angles, lb=0.025, ub=0.975, ax=None):
    """Fit von-Mises and normal distribution on the PHI data"""

    kappa, mean_, scale = vonmises.fit(np.radians(angles), fscale=1)
    mean_norm, std_norm = norm.fit(np.radians(angles))

    print('-------------------')
    print('Fitted von Mises:')
    print(f'Kappa: {kappa:.2f}')
    print(f'Mean: {np.degrees(mean_):.2f}°')

    print('-------------------')
    print('Fitted Normal:')
    print(f'Mean (µ): {np.degrees(mean_norm):.2f}°')
    print(f'Std (σ): {np.degrees(std_norm):.2f}°')

    vals = vonmises.ppf([0.025, 0.5, 0.975], kappa, mean_)
    print('-------------------')
    print('Cut to consider outliers:')
    print(f'Lower bound ({lb:.2f}): {np.degrees(vals[0]):.2f}°')
    print(f'Upper bound ({ub:.2f}): {np.degrees(vals[2]):.2f}°')

    if ax is None:
        fig, ax = plt.subplots(2, 1)

    x = np.linspace(vonmises.ppf(0.001, kappa, mean_), vonmises.ppf(0.999, kappa, mean_), 500)
    vonmises_pdf = vonmises.pdf(x, kappa, mean_)
    max_vonmises_pdf = vonmises_pdf.max()

    x_norm = np.linspace(norm.ppf(0.001, mean_norm, std_norm), norm.ppf(0.999, mean_norm, std_norm), 500)
    norm_pdf = norm.pdf(x_norm, mean_, std_norm)
    max_norm_pdf = norm_pdf.max()

    sns.kdeplot(x=angles, ax=ax)
    sns.rugplot(x=angles, ax=ax)

    max_kde_value = ax.get_lines()[0].get_data()[1].max()
    ax.plot(np.degrees(x),
            vonmises_pdf * (max_kde_value / max_vonmises_pdf), 'r-', lw=0.5, label='Von Mises p.d.f')
    ax.plot(np.degrees(x),
            norm_pdf * (max_kde_value / max_norm_pdf), 'b-', lw=0.5, label='Normal p.d.f')
    ax.set_ylim([0, ax.get_ylim()[1]])
    # ax.set_title('Distributions on the PHI (Azimuthal angle)')
    ax.set_ylabel('Normalized density [-]')
    ax.set_xlabel('Degrees [°]')
    ax.tick_params(axis='y', which='both', left=False, top=False, labelleft=False)

    cut_off_von_mises_radians = vonmises.ppf([lb, ub], kappa, mean_)
    cut_off_von_mises_degrees = np.degrees(cut_off_von_mises_radians)

    cut_off_normal_radians = norm.ppf([lb, ub], mean_, std_norm)
    cut_off_normal_degrees = np.degrees(cut_off_normal_radians)

    ax.axvline(cut_off_von_mises_degrees[0], color='r', linestyle='--', linewidth=0.8, label='Cut-off Von Misses')
    ax.axvline(cut_off_von_mises_degrees[1], color='r', linestyle='--', linewidth=0.8)

    ax.axvline(cut_off_normal_degrees[0], color='b', linestyle='--', linewidth=0.8, label='Cut-off Normal')
    ax.axvline(cut_off_normal_degrees[1], color='b', linestyle='--', linewidth=0.8)
    ax.legend()

    # Return cut from Normal distribution
    return ax, cut_off_normal_degrees

def fit_radius_distribution(radius, lb=0.025, ub=0.975, ax=None):

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    r_mean, r_std = norm.fit(radius)
    x = np.linspace(norm.ppf(0.0001, r_mean, r_std), norm.ppf(0.9999, r_mean, r_std), 1000)
    norm_pdf = norm.pdf(x, r_mean, r_std)
    max_norm_pdf = norm_pdf.max()

    # sns.distplot(radius, hist=False, rug=True, ax=ax)
    sns.kdeplot(x=radius, ax=ax)
    sns.rugplot(x=radius, ax=ax)

    max_kde_value = ax.get_lines()[0].get_data()[1].max()

    ax.plot(x, norm_pdf * (max_kde_value / max_norm_pdf), 'b-', lw=0.5, label='Normal p.d.f')
    ax.set_ylim([0, ax.get_ylim()[1]])
    ax.tick_params(axis='y', which='both', left=False, top=False, labelleft=False)

    cut_off_radius_normal = norm.ppf([lb, ub], r_mean, r_std)
    ax.axvline(cut_off_radius_normal[0], color='b', linestyle='--', linewidth=0.8, label='Cut-off Normal')
    ax.axvline(cut_off_radius_normal[1], color='b', linestyle='--', linewidth=0.8)

    return ax, cut_off_radius_normal

def distributions_and_outliers(r_pca,
                               phi_pca_degrees,
                               theta_pca_degrees,
                               lb,
                               ub,
                               ax=None):

    if ax is None:
        fig, ax = plt.subplots(1, 4, figsize=(13, 4))
        plt.subplots_adjust(bottom=0.15, wspace=0.3, left=0.05, right=0.95)

    ax1, cut_off_radius = fit_radius_distribution(r_pca, lb=lb['radius'], ub=ub['radius'], ax=ax[0])
    ax1.set_title('Dist. Radius')

    ax2, cut_off_phi =  fit_angle_distributions(phi_pca_degrees, lb=lb['phi'], ub=ub['phi'], ax=ax[1])
    ax2.set_title('Dist. PHI (Azimuthal angle)')
    ax2.set_xlim([0, 360])

    ax3, cut_off_theta =  fit_angle_distributions(theta_pca_degrees, lb=lb['theta'], ub=ub['theta'], ax=ax[2])
    ax3.set_title('Dist. THETA (Polar angle)')
    ax3.set_xlim([0, 180])

    idx_radius = (r_pca < cut_off_radius[0]) | (r_pca > cut_off_radius[1])
    idx_phi = (phi_pca_degrees < cut_off_phi[0]) | (phi_pca_degrees > cut_off_phi[1])
    idx_theta = (theta_pca_degrees < cut_off_theta[0]) | (theta_pca_degrees > cut_off_theta[1])

    idx_outliers = idx_radius | idx_phi | idx_theta

    sns.scatterplot(x=phi_pca_degrees[idx_outliers],
                    y=theta_pca_degrees[idx_outliers], ax=ax[3], marker='x', label='Outlier')
    sns.scatterplot(x=phi_pca_degrees[idx_radius],
                    y=theta_pca_degrees[idx_radius], ax=ax[3], marker='x', color='r', label='Outlier radius')
    sns.scatterplot(x=phi_pca_degrees[~idx_outliers],
                    y=theta_pca_degrees[~idx_outliers], ax=ax[3], marker='o')
    ax[3].set_xlim([0, 360])
    ax[3].set_ylim([0, 180])
    ax[3].axvline(cut_off_phi[0], color='r', linestyle='--', linewidth=0.8)
    ax[3].axvline(cut_off_phi[1], color='r', linestyle='--', linewidth=0.8)

    ax[3].axhline(cut_off_theta[0], color='b', linestyle='--', linewidth=0.8)
    ax[3].axhline(cut_off_theta[1], color='b', linestyle='--', linewidth=0.8)

    ax[3].set_xlabel('Phi - Azimuth')
    ax[3].set_ylabel('Theta - Polar')
    ax[3].legend()

    return ax, idx_outliers

def plotly_outlier_visualization(pca_data_clean, pca_data_outliers, file_name):
    # Highlight the outliers from the data
    trace_clean = go.Scatter3d(x=pca_data_clean['PC0'],
                               y=pca_data_clean['PC1'],
                               z=pca_data_clean['PC2'],
                               mode='markers',
                               text=pca_data_clean['DALIBOX_ID'],
                               opacity=0.7,
                               # legendgroup="group" + str(cluster_number),
                               name="Clean",
                               showlegend=True,
                               marker=dict(size=5,
                                           color='#4E79A7')
                               # scene=dict(xaxis=dict(title_text="xxxxxxxxx"))
                               )

    trace_outliers = go.Scatter3d(x=pca_data_outliers['PC0'],
                                  y=pca_data_outliers['PC1'],
                                  z=pca_data_outliers['PC2'],
                                  mode='markers',
                                  text=pca_data_outliers['DALIBOX_ID'],
                                  opacity=0.7,
                                  # legendgroup="group" + str(cluster_number),
                                  name="Outliers",
                                  showlegend=True,
                                  marker=dict(size=5,
                                              color='#F28E2B',
                                              symbol='x')
                                  # scene=dict(xaxis=dict(title_text="xxxxxxxxx"))
                                  )

    fig = go.Figure(data=[trace_clean, trace_outliers])
    fig.update_layout({'scene1': dict(xaxis=dict(title_text="PCA0"),
                                      yaxis=dict(title_text="PCA1"),
                                      zaxis=dict(title_text="PCA2"))},
                      title_text=f"{file_name}",
                      title_x=0.5
                      )
    fig.write_html(f'processed_data/plots/outlier_detection/{file_name}.html', auto_open=True)

def plotly_sphere(coords_pca: np.ndarray,
                  surface_sphere: Tuple[np.ndarray, np.ndarray, np.ndarray],
                  file_name: str) -> None:
    """Make a plotly plot overlaying the surface of a perfect sphere and the data"""
    trace1 = go.Scatter3d(x=coords_pca[:, 0],
                          y=coords_pca[:, 1],
                          z=coords_pca[:, 2],
                          mode='markers',
                          marker=dict(size=5, color='red'))
    trace2 = go.Surface(x=surface_sphere[0],
                        y=surface_sphere[1],
                        z=surface_sphere[2],
                        opacity=0.3, showscale=False, surfacecolor=np.full(surface_sphere[2].shape, 1))

    fig = go.Figure(data=[trace1, trace2])
    fig.update_layout({'scene1': dict(xaxis=dict(title_text="PCA0"),
                                      yaxis=dict(title_text="PCA1"),
                                      zaxis=dict(title_text="PCA2"))},
                      title_text='Fitting of the Sphere',
                      title_x=0.5
                      )
    fig.write_html(f'processed_data/plots/outlier_detection/{file_name}.html', auto_open=True)

    return

def normalize_angle_positive(angle):
    """
    Taken from: mushroom_rl

    Wrap the angle between 0 and 2 * pi.

    Args:
        angle (float): angle to wrap.

    Returns:
         The wrapped angle.

    """
    pi_2 = 2. * np.pi

    return fmod(fmod(angle, pi_2) + pi_2, pi_2)


def normalize_angle(angle):
    """
    Taken from: mushroom_rl

    Wrap the angle between -pi and pi.

    Args:
        angle (float): angle to wrap.

    Returns:
         The wrapped angle.

    """
    a = normalize_angle_positive(angle)
    if a > np.pi:
        a -= 2. * np.pi

    return a

def rotate_angle_degrees(angles: np.ndarray, *, delta_degrees: float):
    assert -180 <= delta_degrees <= 180, "Rotation should be bounded between [-180°, 180°]"

    angle_rad_rotated = np.deg2rad(angles + delta_degrees)
    angle_rad_rotated_wrapped = np.array([normalize_angle_positive(angle_) for angle_ in angle_rad_rotated])
    angle_deg_rotated_wrapped = np.rad2deg(angle_rad_rotated_wrapped)

    return angle_deg_rotated_wrapped



