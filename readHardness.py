import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.image as mpimg
import numpy.typing as npt
from scipy.interpolate import NearestNDInterpolator
from alphashape import alphashape
import geopandas as gpd
from shapely.geometry import Point
from shapely.geometry import Polygon
from scipy.stats import sem


# Imports the hardness data from the Struers Duramin-40 that is exported in .csv format.
#
# Parameters:
#   filename (str): The name of the file to be imported. Must include the file extension.
#
#   x_col (int): The column for the x coordinate data. The default is column 15
#
#   y_col (int): The column for the y coordinate data. The default is column 16
#
#   hardness_col (int): The column for the hardness data. The default is column 19
#
#   delimiter (str): The delimiter seperating the data within the csv. Default value is ','
#
#   overwrite_file (str): The file that contains points that will overwrite data in the original file.
#
#   overwrite_points (list[int]): The points that will be overwritten by the data in the overwrite_file.
#
# Returns:
#   main_data (np.array): The returned x, y, and hardness data from the hardness data file

def hardness_import(filename: str, x_col: int = 15, y_col: int = 16, hardness_col: int = 19, delimiter: str = ',',
                    overwrite_file: str = None, overwrite_points: list[int] = None) -> npt.ArrayLike:
    main_data = np.loadtxt(filename, skiprows=2, delimiter=delimiter, usecols=(x_col, y_col, hardness_col))
    overwrite_data = []
    if overwrite_file is not None:
        overwrite_data = np.loadtxt(overwrite_file, skiprows=2, delimiter=delimiter,
                                    usecols=(x_col, y_col, hardness_col))
    if overwrite_points is not None:
        main_data[overwrite_points, 2] = overwrite_data[:, 2]

    return main_data

# Internal function used to return the hardness data in an organized 2D matrix for the X, Y and Z (hardness) components,
# as well as the exterior xy coordinates that bound the shape of the hardness data.
#
# Parameters
#   data (np.array): The hardness data acquired from the hardness import function.
#
# Returns:
#   XX (np.array): The 2d matrix of the x coordinates of the hardness data
#
#   YY (np.array): The 2d matrix of the y coordinates of the hardness data
#
#   Z (np.array): The 2d matrix of the hardness values from the hardness data
#
#   axy (np.array): The coordinates in x,y of the exterior edge of the hardness data


def xyz_return(data: npt.ArrayLike) -> npt.ArrayLike:

    x, y, z = data.T
    x -= min(x)
    y -= min(y)
    xy = data[:, [0, 1]].copy()
    print(max(x) - min(x), max(y) - min(y))

    interp = NearestNDInterpolator((x, y), z)
    X = np.linspace(min(x), max(x), num=30)
    Y = np.linspace(min(y), max(y), num=50)
    XX, YY = np.meshgrid(X, Y)
    XY = np.dstack((XX, YY))
    Z = np.apply_along_axis(interp, 2, XY)
    Z = Z.reshape(Z.shape[:-1])
    alpha = alphashape(xy[:, :], alpha=1.0)
    axy = alpha.exterior.coords.xy
    axy = list(zip(*axy))

    return XX, YY, Z, axy

# plots the hardness data in an XY colormap. The function expects the data from the hardness_input function, and an axes
# from matplotlib
#
# Parameters:
#   data (np.array): The hardness data acquired from the hardness import function.
#
#   axis (plt.Axes): The matplotlib axis for which will be used to generate the plot.
#
#   hardness_limits (tuple[float, float]): An optional parameter to define the minimum and maximum values of the
#       hardness
#
# Returns:
#   axis (plt.Axes): The axis containing the plotted colormap of the hardness data.
#
#   foo (mpimg.AxesImage): The colormap image attached to the axis. This parameter is useful for setting up a colorbar.


def plot_hardness_data(data: npt.ArrayLike, axis: plt.Axes,
                       hardness_limits: tuple[float, float] = None,) -> (plt.Axes, mpimg.AxesImage):

    x, y, z = data.T
    XX, YY, Z, axy = xyz_return(data)

    patch = mpl.patches.Polygon(axy, transform=axis.transData, facecolor='none', edgecolor='k')
    axis.add_patch(patch)
    axis.set_clip_path(patch)
    foo = axis.imshow(Z, origin='lower', extent=(min(x), max(x), min(y), max(y)))
    foo.set_clip_path(patch)
    if hardness_limits is not None:
        foo.set_clim(hardness_limits[0], hardness_limits[1])
    axis.set_xlim(min(x), max(x))
    axis.set_ylim(min(y), max(y))

    return axis, foo

# The following plots the average hardness over the y-axis of the data.
#
# Parameters:
#   data (np.array): The hardness data acquired from the hardness import function.
#
#   axis (plt.Axes): The matplotlib axis for which will be used to generate the plot.
#
#   label (str): The label to be given to the data.
#
#   hardness_limits (tuple[float, float]): An optional parameter to define the minimum and maximum values of the
#       hardness
#
# Returns:
#   axis (plt.Axes): The axis containing the plotted colormap of the hardness data.


def plot_z_data(data: npt.ArrayLike, axis: plt.Axes, label: str,
                hardness_limits: tuple[float, float] = None) -> plt.Axes:

    x, y, z = data.T
    XX, YY, Z, axy = xyz_return(data)

    geom = [Point(x) for x in np.column_stack((XX.flatten(), YY.flatten()))]
    geo_data = gpd.GeoDataFrame(data=Z.flatten(), geometry=geom)

    mask = Polygon(axy)
    clipped_data = geo_data.clip(mask)

    clipped_data = np.array(clipped_data)
    clipped_z = clipped_data[:, 0]
    clipped_y = np.array([x.y for x in clipped_data[:, 1]])
    clipped_x = np.array([x.x for x in clipped_data[:, 1]])
    clipped_data = np.column_stack((clipped_x, clipped_y, clipped_z))

    adjust = min(y)

    average_h_z = np.array(
        [(np.average(clipped_data[clipped_data[:, 1] == x][:, 2]), x - adjust) for x in np.unique(clipped_y)])
    std_h_z = np.array([sem(clipped_data[clipped_data[:, 1] == x][:, 2]) for x in np.unique(clipped_y)])
    # print(average_h_z[:,0])
    # print(std_h_z)

    bar = [index % 2 == 0 for index, x in enumerate(average_h_z)]
    baz = [(index + 1) % 2 == 0 for index, x in enumerate(average_h_z)]

    axis.errorbar(average_h_z[:, 1], average_h_z[:, 0], yerr=std_h_z, uplims=bar, lolims=baz, label=label)
    if hardness_limits is not None:
        axis.set_ylim(hardness_limits[0], hardness_limits[1])

    return axis


# The following code below demonstrates the plotting of colormaps of the hardness data. If this file is ran directly the
# following code below will run.
if __name__ == '__main__':
    data_demo_1 = hardness_import("Demo_File_1.csv")
    data_demo_2 = hardness_import("Demo_File_2.csv")

    fig = plt.figure(dpi=500, figsize=(7, 4))
    gs = fig.add_gridspec(1, 2, wspace=0.1, hspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    ax, _ = plot_hardness_data(data_demo_1, ax, (300, 420))
    ax.set_title("Demo 1")
    ax.set_ylabel("z (mm)")
    ax.set_xlabel("x (mm)")

    ax = fig.add_subplot(gs[0, 1])
    ax, foo = plot_hardness_data(data_demo_2, ax, (300, 420))
    ax.set_title("Demo 2")
    ax.set_xlabel("x (mm)")

    cbar_axes = [0.9, 0.1, 0.02, 0.8]
    cbar_ax = fig.add_axes(cbar_axes)
    fig.colorbar(foo, cax=cbar_ax, label="Hardness (HV1)")

    plt.show()
