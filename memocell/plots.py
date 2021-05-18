

import numpy as np
# from networkx.drawing.nx_agraph import to_agraph
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
import corner
from cycler import cycler
import os
from IPython.display import Image, display

from .selection import _dots_w_bars_evidence, _dots_wo_bars_likelihood_max, _dots_wo_bars_bic, _dots_wo_bars_evidence_from_bic
import warnings
# NOTE: mark as comment for cluster computations
# import graphviz

# dynesty plotting utilities
from dynesty import plotting as dyplot

__all__ = ["net_main_plot", "net_hidden_plot",
            "sim_counts_plot", "sim_mean_plot", "sim_variance_plot", "sim_covariance_plot",
            "data_mean_plot", "data_variance_plot", "data_covariance_plot",
            "data_hist_variables_plot", "data_hist_waiting_times_plot", "data_variable_scatter_plot",
            "selection_plot", "est_runplot", "est_traceplot",
            "est_parameter_plot", "est_corner_plot", "est_corner_kernel_plot",
            "est_corner_weight_plot", "est_chains_plot", # "est_corner_bounds_plot",
            "est_bestfit_mean_plot", "est_bestfit_variance_plot", "est_bestfit_covariance_plot"]

###############################
### MAIN PLOTTING FUNCTIONS ###
###############################

def net_main_plot(net, node_settings=None, edge_settings=None,
                                            layout=None, show=True, save=None):
    """Plot the main layer of a network.

    Parameters
    ----------
    net : memocell.network.Network
        A memocell network object.
    node_settings : dict of dict, optional
        Optional label and color settings for the network nodes. Colors
        require hex literals and labels require html (not latex) format.
    edge_settings : dict of dict, optional
        Optional label and color settings for the network edges. Colors
        require hex literals and labels require html (not latex) format.
    layout : None or str, optional
        Specify layout engine for computing node positions; e.g. 'dot', 'neato', 'circo'.
    show : bool, optional
        Network plot is shown if `show=True`.
    save : None or str, optional
        Provide a path to save the plot.

    Returns
    -------
    pdot : pydot.Dot
        Pydot network main layer object.
    """

    # if not given, create some default node settings
    if node_settings==None:
        nodes = list(net.net_nodes_identifier.values())
        colors = sns.color_palette('Set2', n_colors=len(nodes)).as_hex()
        node_settings = dict()

        for i, node in enumerate(nodes):
            node_settings[node] = {'label': node, 'color': colors[i]}

    # if not given, create some default edge settings
    if edge_settings==None:
        edge_settings = dict()
        for rate in net.net_rates_identifier.values():
            edge_settings[rate] = {'label': rate, 'color': 'black'}

    # create net object with plotting info for graphviz
    net_graphviz, layout_engine = net._draw_main_network_graph(node_settings, edge_settings)
    pdot = nx.drawing.nx_pydot.to_pydot(net_graphviz)

    # overwrite default layout_engine if specified
    if layout!=None:
        layout_engine = layout

    # save/show figure
    if save!=None:
        pdot.write_pdf(save, prog=layout_engine)

    if show:
        display(Image(pdot.create_png(prog=layout_engine)))
    return pdot


def net_hidden_plot(net, node_settings=None, edge_settings=None,
                                            layout=None, show=True, save=None):
    """Plot the hidden layer of a network.

    Parameters
    ----------
    net : memocell.network.Network
        A memocell network object.
    node_settings : dict of dict, optional
        Optional label and color settings for the network nodes. Colors
        require hex literals and labels require html (not latex) format.
    edge_settings : dict of dict, optional
        Optional label and color settings for the network edges. Colors
        require hex literals and labels require html (not latex) format.
    layout : None or str, optional
        Specify layout engine for computing node positions; e.g. 'dot', 'neato', 'circo'.
    show : bool, optional
        Network plot is shown if `show=True`.
    save : None or str, optional
        Provide a path to save the plot.

    Returns
    -------
    pdot : pydot.Dot
        Pydot network hidden layer object.
    """

    # if not given, create some default node settings
    if node_settings==None:
        nodes = list(net.net_nodes_identifier.values())
        colors = sns.color_palette('Set2', n_colors=len(nodes)).as_hex()
        node_settings = dict()

        for i, node in enumerate(nodes):
            node_settings[node] = {'label': node, 'color': colors[i]}

    # if not given, create some default edge settings
    if edge_settings==None:
        edge_settings = dict()
        for rate in net.net_rates_identifier.values():
            edge_settings[rate] = {'label': rate, 'color': 'black'}

    # create net object with plotting info for graphviz
    net_graphviz, layout_engine = net._draw_hidden_network_graph(node_settings, edge_settings)
    pdot = nx.drawing.nx_pydot.to_pydot(net_graphviz)

    # overwrite default layout_engine if specified
    if layout!=None:
        layout_engine = layout

    # save/show figure
    if save!=None:
        pdot.write_pdf(save, prog=layout_engine)

    if show:
        display(Image(pdot.create_png(prog=layout_engine)))
    return pdot


def sim_counts_plot(sim, settings=None,
                    xlabel='Time', xlim=None, xlog=False,
                    ylabel='Counts', ylim=None, ylog=False,
                    show=True, save=None):
    """Plot the variable count for a given simulation.

    Parameters
    ----------
    sim : memocell.simulation.Simulation
        A memocell simulation object that contains a Gillespie simulation.
    settings : dict of dict, optional
        Optional label and color settings for the simulation variables.
    xlabel : str, optional
        Label for x-axis.
    xlim : None or tuple of floats, optional
        Specify x-axis limits.
    xlog : bool, optional
        Logarithmic x-axis if `xlog=True`.
    ylabel : str, optional
        Label for y-axis.
    ylim : None or tuple of floats, optional
        Specify y-axis limits.
    ylog : bool, optional
        Logarithmic y-axis if `ylog=True`.
    show : bool, optional
        Plot is shown if `show=True`.
    save : None or str, optional
        Provide a path to save the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : list or array of matplotlib.axes
    """

    # if not given, create some default settings
    if settings==None:
        vars = [sim.sim_variables_identifier[var[0]][0] for var in sim.sim_variables_order[0]]
        colors = sns.color_palette('Set2', n_colors=len(vars)).as_hex()
        settings = dict()

        for i, var in enumerate(vars):
            settings[var] = {'label': var, 'color': colors[i]}

    # create plotting information from simulation
    x_arr, y_arr, attributes = sim._line_evolv_counts(settings)

    # plot by line_evolv utility plotting function
    fig, axes = _line_evolv(x_arr, y_arr, attributes,
                    xlabel=xlabel, xlim=xlim, xlog=xlog,
                    ylabel=ylabel, ylim=ylim, ylog=ylog,
                    show=show, save=save)
    return fig, axes


def sim_mean_plot(sim, settings=None,
                    xlabel='Time', xlim=None, xlog=False,
                    ylabel='Mean', ylim=None, ylog=False,
                    show=True, save=None):
    """Plot the mean (or expectation) dynamics of a given simulation.

    Parameters
    ----------
    sim : memocell.simulation.Simulation
        A memocell simulation object that contains a moment simulation.
    settings : dict of dict, optional
        Optional label and color settings for the mean traces.
    xlabel : str, optional
        Label for x-axis.
    xlim : None or tuple of floats, optional
        Specify x-axis limits.
    xlog : bool, optional
        Logarithmic x-axis if `xlog=True`.
    ylabel : str, optional
        Label for y-axis.
    ylim : None or tuple of floats, optional
        Specify y-axis limits.
    ylog : bool, optional
        Logarithmic y-axis if `ylog=True`.
    show : bool, optional
        Plot is shown if `show=True`.
    save : None or str, optional
        Provide a path to save the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : list or array of matplotlib.axes
    """

    # if not given, create some default settings
    if settings==None:
        vars = [sim.sim_variables_identifier[var[0]][0] for var in sim.sim_variables_order[0]]
        colors = sns.color_palette('Set2', n_colors=len(vars)).as_hex()
        settings = dict()

        for i, var in enumerate(vars):
            settings[var] = {'label': f'E({var})', 'color': colors[i]}

    # create plotting information from simulation
    x_arr, y_arr, attributes = sim._line_evolv_mean(settings)

    # plot by line_evolv utility plotting function
    fig, axes = _line_evolv(x_arr, y_arr, attributes,
                    xlabel=xlabel, xlim=xlim, xlog=xlog,
                    ylabel=ylabel, ylim=ylim, ylog=ylog,
                    show=show, save=save)
    return fig, axes


def sim_variance_plot(sim, settings=None,
                    xlabel='Time', xlim=None, xlog=False,
                    ylabel='Variance', ylim=None, ylog=False,
                    show=True, save=None):
    """Plot the variance dynamics of a given simulation.

    Parameters
    ----------
    sim : memocell.simulation.Simulation
        A memocell simulation object that contains a moment simulation.
    settings : dict of dict, optional
        Optional label and color settings for the variance traces.
    xlabel : str, optional
        Label for x-axis.
    xlim : None or tuple of floats, optional
        Specify x-axis limits.
    xlog : bool, optional
        Logarithmic x-axis if `xlog=True`.
    ylabel : str, optional
        Label for y-axis.
    ylim : None or tuple of floats, optional
        Specify y-axis limits.
    ylog : bool, optional
        Logarithmic y-axis if `ylog=True`.
    show : bool, optional
        Plot is shown if `show=True`.
    save : None or str, optional
        Provide a path to save the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : list or array of matplotlib.axes
    """

    # if not given, create some default settings
    if settings==None:
        vars = [(sim.sim_variables_identifier[var[0]][0], sim.sim_variables_identifier[var[1]][0])
                    for var in sim.sim_variables_order[1] if var[0]==var[1]]
        colors = sns.color_palette('Set2', n_colors=len(vars)).as_hex()
        settings = dict()

        for i, var in enumerate(vars):
            settings[var] = {'label': f'Var({var[0]})', 'color': colors[i]}

    # create plotting information from simulation
    x_arr, y_arr, attributes = sim._line_evolv_variance(settings)

    # plot by line_evolv utility plotting function
    fig, axes = _line_evolv(x_arr, y_arr, attributes,
                    xlabel=xlabel, xlim=xlim, xlog=xlog,
                    ylabel=ylabel, ylim=ylim, ylog=ylog,
                    show=show, save=save)
    return fig, axes


def sim_covariance_plot(sim, settings=None,
                    xlabel='Time', xlim=None, xlog=False,
                    ylabel='Covariance', ylim=None, ylog=False,
                    show=True, save=None):
    """Plot the covariance dynamics of a given simulation.

    Parameters
    ----------
    sim : memocell.simulation.Simulation
        A memocell simulation object that contains a moment simulation.
    settings : dict of dict, optional
        Optional label and color settings for the covariance traces.
    xlabel : str, optional
        Label for x-axis.
    xlim : None or tuple of floats, optional
        Specify x-axis limits.
    xlog : bool, optional
        Logarithmic x-axis if `xlog=True`.
    ylabel : str, optional
        Label for y-axis.
    ylim : None or tuple of floats, optional
        Specify y-axis limits.
    ylog : bool, optional
        Logarithmic y-axis if `ylog=True`.
    show : bool, optional
        Plot is shown if `show=True`.
    save : None or str, optional
        Provide a path to save the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : list or array of matplotlib.axes
    """

    # if not given, create some default settings
    if settings==None:
        vars = [(sim.sim_variables_identifier[var[0]][0], sim.sim_variables_identifier[var[1]][0])
                    for var in sim.sim_variables_order[1] if var[0]!=var[1]]
        colors = sns.color_palette('Set2', n_colors=len(vars)).as_hex() # 'muted'
        settings = dict()

        for i, var in enumerate(vars):
            settings[var] = {'label': f'Cov({var[0]}, {var[1]})', 'color': colors[i]}

    # create plotting information from simulation
    x_arr, y_arr, attributes = sim._line_evolv_covariance(settings)

    # plot by line_evolv utility plotting function
    fig, axes = _line_evolv(x_arr, y_arr, attributes,
                    xlabel=xlabel, xlim=xlim, xlog=xlog,
                    ylabel=ylabel, ylim=ylim, ylog=ylog,
                    show=show, save=save)
    return fig, axes


def data_mean_plot(data, settings=None,
                    xlabel='Time', xlim=None, xlog=False,
                    ylabel='Mean', ylim=None, ylog=False,
                    show=True, save=None):
    """Plot the mean statistics with standard errors of a data object.

    Parameters
    ----------
    data : memocell.data.Data
        A memocell data object.
    settings : dict of dict, optional
        Optional label and color settings for the mean traces.
    xlabel : str, optional
        Label for x-axis.
    xlim : None or tuple of floats, optional
        Specify x-axis limits.
    xlog : bool, optional
        Logarithmic x-axis if `xlog=True`.
    ylabel : str, optional
        Label for y-axis.
    ylim : None or tuple of floats, optional
        Specify y-axis limits.
    ylog : bool, optional
        Logarithmic y-axis if `ylog=True`.
    show : bool, optional
        Plot is shown if `show=True`.
    save : None or str, optional
        Provide a path to save the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : list or array of matplotlib.axes
    """

    # if not given, create some default settings
    if settings==None:
        vars = [d['variables'] for d in data.data_mean_order]
        colors = sns.color_palette('Set2', n_colors=len(vars)).as_hex()
        settings = dict()

        for i, var in enumerate(vars):
            settings[var] = {'label': f'E({var})', 'color': colors[i]}

    # create plotting information from data
    x_arr, y_arr, attributes = data._dots_w_bars_evolv_mean(settings)

    # plot by dots_w_bars_evolv utility plotting function
    fig, axes = _dots_w_bars_evolv(x_arr, y_arr, attributes,
                    xlabel=xlabel, xlim=xlim, xlog=xlog,
                    ylabel=ylabel, ylim=ylim, ylog=ylog,
                    show=show, save=save)
    return fig, axes


def data_variance_plot(data, settings=None,
                    xlabel='Time', xlim=None, xlog=False,
                    ylabel='Variance', ylim=None, ylog=False,
                    show=True, save=None):
    """Plot the variance statistics with standard errors of a data object (if
    variance data are available).

    Parameters
    ----------
    data : memocell.data.Data
        A memocell data object.
    settings : dict of dict, optional
        Optional label and color settings for the variance traces.
    xlabel : str, optional
        Label for x-axis.
    xlim : None or tuple of floats, optional
        Specify x-axis limits.
    xlog : bool, optional
        Logarithmic x-axis if `xlog=True`.
    ylabel : str, optional
        Label for y-axis.
    ylim : None or tuple of floats, optional
        Specify y-axis limits.
    ylog : bool, optional
        Logarithmic y-axis if `ylog=True`.
    show : bool, optional
        Plot is shown if `show=True`.
    save : None or str, optional
        Provide a path to save the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : list or array of matplotlib.axes
    """

    # if not given, create some default settings
    if settings==None:
        vars = [d['variables'] for d in data.data_variance_order]
        colors = sns.color_palette('Set2', n_colors=len(vars)).as_hex()
        settings = dict()

        for i, var in enumerate(vars):
            settings[var] = {'label': f'Var({var[0]})', 'color': colors[i]}

    # create plotting information from data
    x_arr, y_arr, attributes = data._dots_w_bars_evolv_variance(settings)

    # plot by dots_w_bars_evolv utility plotting function
    fig, axes = _dots_w_bars_evolv(x_arr, y_arr, attributes,
                    xlabel=xlabel, xlim=xlim, xlog=xlog,
                    ylabel=ylabel, ylim=ylim, ylog=ylog,
                    show=show, save=save)
    return fig, axes


def data_covariance_plot(data, settings=None,
                    xlabel='Time', xlim=None, xlog=False,
                    ylabel='Covariance', ylim=None, ylog=False,
                    show=True, save=None):
    """Plot the covariance statistics with standard errors of a data object (if
    covariance data are available).

    Parameters
    ----------
    data : memocell.data.Data
        A memocell data object.
    settings : dict of dict, optional
        Optional label and color settings for the covariance traces.
    xlabel : str, optional
        Label for x-axis.
    xlim : None or tuple of floats, optional
        Specify x-axis limits.
    xlog : bool, optional
        Logarithmic x-axis if `xlog=True`.
    ylabel : str, optional
        Label for y-axis.
    ylim : None or tuple of floats, optional
        Specify y-axis limits.
    ylog : bool, optional
        Logarithmic y-axis if `ylog=True`.
    show : bool, optional
        Plot is shown if `show=True`.
    save : None or str, optional
        Provide a path to save the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : list or array of matplotlib.axes
    """

    # if not given, create some default settings
    if settings==None:
        vars = [d['variables'] for d in data.data_covariance_order]
        colors = sns.color_palette('Set2', n_colors=len(vars)).as_hex()
        settings = dict()

        for i, var in enumerate(vars):
            settings[var] = {'label': f'Cov({var[0]}, {var[1]})', 'color': colors[i]}

    # create plotting information from data
    x_arr, y_arr, attributes = data._dots_w_bars_evolv_covariance(settings)

    # plot by dots_w_bars_evolv utility plotting function
    fig, axes = _dots_w_bars_evolv(x_arr, y_arr, attributes,
                    xlabel=xlabel, xlim=xlim, xlog=xlog,
                    ylabel=ylabel, ylim=ylim, ylog=ylog,
                    show=show, save=save)
    return fig, axes


def data_hist_variables_plot(data, time_ind, normalised=False, settings=None,
                            xlabel='Variable counts', xlim=None, xlog=False,
                            ylabel='Frequency', ylim=None, ylog=False,
                            show=True, save=None):
    """Histograms of variable counts of a data object at a given time point.

    Parameters
    ----------
    data : memocell.data.Data
        A memocell data object.
    time_ind : int
        Time index to take variable counts from; the time point is then given by
        `data.data_time_values[time_ind]`.
    normalised : bool, optional
        Histograms are normalised if `normalised=True`.
    settings : dict of dict, optional
        Optional label, color and opacity settings for histogram variables.
    xlabel : str, optional
        Label for x-axis.
    xlim : None or tuple of floats, optional
        Specify x-axis limits.
    xlog : bool, optional
        Logarithmic x-axis if `xlog=True`.
    ylabel : str, optional
        Label for y-axis.
    ylim : None or tuple of floats, optional
        Specify y-axis limits.
    ylog : bool, optional
        Logarithmic y-axis if `ylog=True`.
    show : bool, optional
        Plot is shown if `show=True`.
    save : None or str, optional
        Provide a path to save the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : list or array of matplotlib.axes
    """

    # if not given, create some default settings
    if settings==None:
        vars = [d for d in data.data_variables]
        colors = sns.color_palette('Set2', n_colors=len(vars)).as_hex()
        settings = dict()

        for i, var in enumerate(vars):
            settings[var] = {'label': var, 'color': colors[i], 'opacity': 0.5}

    # create plotting information from data
    bar_arr, bar_attributes = data._histogram_discrete_cell_counts_at_time_point(time_ind, settings)

    # plot by histogram_discrete utility plotting function
    fig, axes = _histogram_discrete(bar_arr, bar_attributes, normalised=normalised,
                            xlabel=xlabel, xlim=xlim, xlog=xlog,
                            ylabel=ylabel, ylim=ylim, ylog=ylog,
                            show=show, save=save)
    return fig, axes


def data_hist_waiting_times_plot(data, data_events, normalised=True,
                            gamma_fit=True, settings=None,
                            xlabel='Waiting time', xlim=None, xlog=False,
                            ylabel='Frequency', ylim=None, ylog=False,
                            show=True, save=None):
    """Histogram of waiting times of a data object for a given event.

    Parameters
    ----------
    data : memocell.data.Data
        A memocell data object.
    data_events : list of tuples with (bool, float or None)
        Data events to take waiting times from; e.g. `data.event_all_first_cell_type_conversion`.
        The `boolean` value of the tuples indicates whether an event happened or not. If not,
        the second value should be `None`. If yes, the second value should be a float of the waiting time
        for that event to happen.
    normalised : bool, optional
        Histograms are normalised if `normalised=True`.
    gamma_fit : bool, optional
        If `gamma_fit=True`, fit a Gamma distribution to binned waiting time data. Fitted shape and scale
        parameters are printed; fitted shape (=steps `n`) and mean waiting time (`theta` = shape * scale)
        are shown in the legend label.
    settings : dict of dict, optional
        Optional label, color, opacity and gamma_color settings for the histogram.
    xlabel : str, optional
        Label for x-axis.
    xlim : None or tuple of floats, optional
        Specify x-axis limits.
    xlog : bool, optional
        Logarithmic x-axis if `xlog=True`.
    ylabel : str, optional
        Label for y-axis.
    ylim : None or tuple of floats, optional
        Specify y-axis limits.
    ylog : bool, optional
        Logarithmic y-axis if `ylog=True`.
    show : bool, optional
        Plot is shown if `show=True`.
    save : None or str, optional
        Provide a path to save the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : list or array of matplotlib.axes
    """

    # if not given, create some default settings
    if settings==None:
        settings = {'label': 'Event waiting times', 'opacity': 1.0,
                    'color': 'orange', 'gamma_color': 'darkorange'}

    # create plotting information from data and
    # plot by respective continuous histogram utility plotting functions
    if gamma_fit:
        normalised = True # overwrite normalised, gamma fit currently only as pdf
        bar_arr, bar_attributes, gamma_fit_func = data._histogram_continuous_event_waiting_times_w_gamma_fit(data_events, settings)
        fig, axes = _histogram_continuous_w_line(bar_arr, bar_attributes, gamma_fit_func, normalised=normalised,
                                xlabel=xlabel, xlim=xlim, xlog=xlog,
                                ylabel=ylabel, ylim=ylim, ylog=ylog,
                                show=show, save=save)
    else:
        bar_arr, bar_attributes = data._histogram_continuous_event_waiting_times(data_events, settings)
        fig, axes = _histogram_continuous(bar_arr, bar_attributes, normalised=normalised,
                                xlabel=xlabel, xlim=xlim, xlog=xlog,
                                ylabel=ylabel, ylim=ylim, ylog=ylog,
                                show=show, save=save)
    return fig, axes


def data_variable_scatter_plot(data, time_ind, variable1, variable2, settings=None,
                                xlabel=None, xlim=None, xlog=False,
                                ylabel=None, ylim=None, ylog=False,
                                show=True, save=None):
    """Scatter plot between counts of two variables of a data object at a given time point.

    Parameters
    ----------
    data : memocell.data.Data
        A memocell data object.
    time_ind : int
        Time index to take variable counts from; the time point is then given by
        `data.data_time_values[time_ind]`.
    variable1 : str
        First variable of the scatter plot (x-axis).
    variable2 : str
        Second variable of the scatter plot (y-axis).
    settings : dict, optional
        Optional label, color and opacity settings for the scatter plot.
    xlabel : str, optional
        Label for x-axis.
    xlim : None or tuple of floats, optional
        Specify x-axis limits.
    xlog : bool, optional
        Logarithmic x-axis if `xlog=True`.
    ylabel : str, optional
        Label for y-axis.
    ylim : None or tuple of floats, optional
        Specify y-axis limits.
    ylog : bool, optional
        Logarithmic y-axis if `ylog=True`.
    show : bool, optional
        Plot is shown if `show=True`.
    save : None or str, optional
        Provide a path to save the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : list or array of matplotlib.axes
    """

    # if not given, create some default settings
    if settings==None:
        settings = {'color': 'orange', 'opacity': 0.5, 'label': None}

    # create plotting information from data
    x_arr, y_arr, attributes = data._scatter_at_time_point(variable1, variable2, time_ind, settings)

    # plot by scatter utility plotting function
    fig, axes = _scatter(x_arr, y_arr, attributes,
                                xlabel=xlabel, xlim=xlim, xlog=xlog,
                                ylabel=ylabel, ylim=ylim, ylog=ylog,
                                show=show, save=save)
    return fig, axes


def selection_plot(estimation_instances, est_type='evidence',
                    settings=None, show_errorbar=True,
                    xlabel=None, xlim=None, xlog=False,
                    ylabel=None, ylim=None, ylog=False,
                    show=True, save=None):
    """Scatter plot between counts of two variables of a data object at a given time point.

    Parameters
    ----------
    estimation_instances : list of memocell.estimation.Estimation
        A list of memocell estimation objects.
    est_type : str, optional
        Specify type of selection plot. `est_type='evidence'` (default) to plot logarithmic
        evidence values from nested sampling. `est_type='likelihood'` to plot the maximal
        logarithmic likelihood value from nested sampling. `est_type='bic'` to plot the
        Bayesian information criterion based on the maximal logarithmic likelihood
        (from nested sampling), number of data points n and number of parameters k
        (BIC = ln(n) k - 2 ln(Lmax)). `est_type='evidence_from_bic'` to plot logarithmic
        evidence values (approximated from the BIC values via evidence ≈ exp(-BIC / 2)).
    settings : dict of dict, optional
        Optional label and color for the estimation instances.
    show_errorbar : bool, optional
        Error bars are shown if `show_errorbar=True` (default). Only available for
        `est_type='evidence'`.
    xlabel : str, optional
        Label for x-axis.
    xlim : None or tuple of floats, optional
        Specify x-axis limits.
    xlog : bool, optional
        Logarithmic x-axis if `xlog=True`.
    ylabel : str, optional
        Label for y-axis.
    ylim : None or tuple of floats, optional
        Specify y-axis limits.
    ylog : bool, optional
        Logarithmic y-axis if `ylog=True`.
    show : bool, optional
        Plot is shown if `show=True`.
    save : None or str, optional
        Provide a path to save the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : list or array of matplotlib.axes
    """

    # if not given, create some default settings
    if settings==None:
        est_names = [est.est_name for est in estimation_instances]
        settings = dict()

        for i, est_name in enumerate(est_names):
            settings[est_name] = {'label': est_name, 'color': 'orange'}

    # create plotting information from estimation_instances
    # afterwards plot by dots_w_bars utility plotting function
    if est_type=='evidence':
        ylabel = 'Log(Evidence)'
        y_arr_err, x_ticks, attributes = _dots_w_bars_evidence(estimation_instances, settings)
        fig, axes = _dots_w_bars(y_arr_err, x_ticks, attributes, show_errorbar=show_errorbar,
                    xlabel=xlabel, xlim=xlim, xlog=xlog,
                    ylabel=ylabel, ylim=ylim, ylog=ylog,
                    show=show, save=save)
    elif est_type=='likelihood':
        ylabel = 'Max. Log(Likelihood)'
        y_arr_err, x_ticks, attributes = _dots_wo_bars_likelihood_max(estimation_instances, settings)
        fig, axes = _dots_w_bars(y_arr_err, x_ticks, attributes, show_errorbar=False,
                    xlabel=xlabel, xlim=xlim, xlog=xlog,
                    ylabel=ylabel, ylim=ylim, ylog=ylog,
                    show=show, save=save)
    elif est_type=='bic':
        ylabel = 'BIC'
        y_arr_err, x_ticks, attributes = _dots_wo_bars_bic(estimation_instances, settings)
        fig, axes = _dots_w_bars(y_arr_err, x_ticks, attributes, show_errorbar=False,
                    xlabel=xlabel, xlim=xlim, xlog=xlog,
                    ylabel=ylabel, ylim=ylim, ylog=ylog,
                    show=show, save=save)
    elif est_type=='evidence_from_bic':
        ylabel = 'Log(Evidence) [approx. from BIC]'
        y_arr_err, x_ticks, attributes = _dots_wo_bars_evidence_from_bic(estimation_instances, settings)
        fig, axes = _dots_w_bars(y_arr_err, x_ticks, attributes, show_errorbar=False,
                    xlabel=xlabel, xlim=xlim, xlog=xlog,
                    ylabel=ylabel, ylim=ylim, ylog=ylog,
                    show=show, save=save)
    else:
        warnings.warn('Unknown est_type for model selection plot.')

    return fig, axes


def est_runplot(estimation, color='limegreen', show=True, save=None):
    """Wrapper to runplot of `dynesty <https://dynesty.readthedocs.io/en/latest/quickstart.html>`_
    nested sampling module.

    Parameters
    ----------
    estimation : memocell.estimation.Estimation
        A memocell estimation object.
    color : str, optional
        Optional color passed to `dynesty`'s `runplot`.
    show : bool, optional
        Plot is shown if `show=True`.
    save : None or str, optional
        Provide a path to save the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : list or array of matplotlib.axes
    """

    # get plotting information from estimation instance
    sampler_result = estimation.bay_nested_sampler_res

    # generate plot by dynesty plotting methods
    fig, axes = dyplot.runplot(sampler_result, color=color)

    # save/show figure
    if save!=None:
        plt.savefig(save, bbox_inches='tight')
    if show:
        plt.show(fig, block=False)
    return fig, axes


def est_traceplot(estimation, settings=None, show=True, save=None):
    """Wrapper to traceplot of `dynesty <https://dynesty.readthedocs.io/en/latest/quickstart.html>`_
    nested sampling module.

    Parameters
    ----------
    estimation : memocell.estimation.Estimation
        A memocell estimation object.
    settings : dict of dict, optional
        Optional labels for parameters.
    show : bool, optional
        Plot is shown if `show=True`.
    save : None or str, optional
        Provide a path to save the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : list or array of matplotlib.axes
    """

    # if not given, create some default settings
    if settings==None:
        settings = dict()
        for theta_id in estimation.net.net_theta_symbolic:
            param = estimation.net.net_rates_identifier[theta_id]
            settings[param] = {'label': param}

    # get plotting information from estimation instance
    sampler_result, params_labels = estimation._sampling_res_and_labels(settings)

    # generate plot by dynesty plotting methods
    fig, axes = dyplot.traceplot(sampler_result,
                         show_titles=True,
                         post_color='dodgerblue',
                         connect_color='darkorange',
                         trace_cmap='magma',
                         connect=True,
                         connect_highlight=range(5),
                         labels=params_labels #, title_fmt='.4f'
                         )

    # save/show figure
    if save!=None:
        plt.savefig(save, bbox_inches='tight')
    if show:
        plt.show(fig, block=False)
    return fig, axes


def est_parameter_plot(estimation, settings=None, show_errorbar=True,
            xlabel=None, xlim=None,
            ylabel="Parameter values", ylim=None, ylog=False,
            show=True, save=None):
    """Plot summary of 1-dimensional marginal parameter posteriors as median and
    95% credible interval (2.5th and 97.5th percentiles; error bars).

    Parameters
    ----------
    estimation : memocell.estimation.Estimation
        A memocell estimation object.
    settings : dict of dict, optional
        Optional label and color for parameters.
    show_errorbar : bool, optional
        Error bars are shown if `show_errorbar=True` (default). Error bars show
        95% credible intervals based on 2.5th and 97.5th percentiles for each
        parameter's 1-dimensional marginal posterior distribution.
    xlabel : str, optional
        Label for x-axis.
    xlim : None or tuple of floats, optional
        Specify x-axis limits.
    ylabel : str, optional
        Label for y-axis.
    ylim : None or tuple of floats, optional
        Specify y-axis limits.
    ylog : bool, optional
        Logarithmic y-axis if `ylog=True`.
    show : bool, optional
        Plot is shown if `show=True`.
    save : None or str, optional
        Provide a path to save the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : list or array of matplotlib.axes
    """

    # if not given, create some default settings
    if settings==None:
        settings = dict()
        for theta_id in estimation.net.net_theta_symbolic:
            param = estimation.net.net_rates_identifier[theta_id]
            settings[param] = {'label': param, 'color': 'dodgerblue'}

    # create plotting information from estimation instance
    y_arr_err, x_ticks, attributes = estimation._dots_w_bars_parameters(settings)

    # plot by dots_w_bars utility plotting function
    fig, axes = _dots_w_bars(y_arr_err, x_ticks, attributes, show_errorbar=show_errorbar,
                xlabel=xlabel, xlim=xlim, xlog=False,
                ylabel=ylabel, ylim=ylim, ylog=ylog,
                show=show, save=save)
    return fig, axes


def est_corner_plot(estimation, settings=None, show=True, save=None):
    """Wrapper to corner plot of `corner <https://corner.readthedocs.io/en/latest/>`_ module;
    visualisation of the parameter posterior distribution by all 2-dimensional and
    1-dimensional marginals.

    Parameters
    ----------
    estimation : memocell.estimation.Estimation
        A memocell estimation object.
    settings : dict of dict, optional
        Optional labels for parameters.
    show : bool, optional
        Plot is shown if `show=True`.
    save : None or str, optional
        Provide a path to save the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : list or array of matplotlib.axes
    """

    # if not given, create some default settings
    if settings==None:
        settings = dict()
        for theta_id in estimation.net.net_theta_symbolic:
            param = estimation.net.net_rates_identifier[theta_id]
            settings[param] = {'label': param}

    # get plotting information from estimation instance
    samples, labels = estimation._samples_corner_parameters(settings)

    # use corner package for this plot
    fig = corner.corner(samples, labels=labels)

    # save/show figure
    if save!=None:
        plt.savefig(save, bbox_inches='tight')
    if show:
        plt.show(fig, block=False)
    return fig, fig.axes


def est_corner_kernel_plot(estimation, settings=None, show=True, save=None):
    """Wrapper to corner plot of `dynesty <https://dynesty.readthedocs.io/en/latest/quickstart.html>`_
    nested sampling module; visualisation of the parameter posterior distribution
    by all 2-dimensional and 1-dimensional marginals (kernel smoothed).

    Parameters
    ----------
    estimation : memocell.estimation.Estimation
        A memocell estimation object.
    settings : dict of dict, optional
        Optional labels for parameters.
    show : bool, optional
        Plot is shown if `show=True`.
    save : None or str, optional
        Provide a path to save the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : (list or array of) matplotlib.axes
    """

    # if not given, create some default settings
    if settings==None:
        settings = dict()
        for theta_id in estimation.net.net_theta_symbolic:
            param = estimation.net.net_rates_identifier[theta_id]
            settings[param] = {'label': param}

    # get plotting information from estimation instance
    sampler_result, params_labels = estimation._sampling_res_and_labels(settings)

    # generate plot by dynesty plotting methods
    fig, axes = dyplot.cornerplot(sampler_result,
                            color='#A0B1BA',
                            show_titles=True,
                            labels=params_labels,
                            title_fmt='.4f')
    fig.tight_layout()

    # save/show figure
    if save!=None:
        plt.savefig(save, bbox_inches='tight')
    if show:
        plt.show(fig, block=False)
    return fig, axes


def est_corner_weight_plot(estimation, settings=None, show=True, save=None):
    """Wrapper to cornerpoints plot of `dynesty <https://dynesty.readthedocs.io/en/latest/quickstart.html>`_
    nested sampling module.

    Parameters
    ----------
    estimation : memocell.estimation.Estimation
        A memocell estimation object.
    settings : dict of dict, optional
        Optional labels for parameters.
    show : bool, optional
        Plot is shown if `show=True`.
    save : None or str, optional
        Provide a path to save the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : (list or array of) matplotlib.axes
    """
    # NOTE: this plot does not seem to work in 1d (dynesty issue, not memocell)

    # if not given, create some default settings
    if settings==None:
        settings = dict()
        for theta_id in estimation.net.net_theta_symbolic:
            param = estimation.net.net_rates_identifier[theta_id]
            settings[param] = {'label': param}

    # get plotting information from estimation instance
    sampler_result, params_labels = estimation._sampling_res_and_labels(settings)

    # generate plot by dynesty plotting methods
    fig, axes = dyplot.cornerpoints(sampler_result,
                         cmap='magma',
                         labels=params_labels)

    # save/show figure
    if save!=None:
        plt.savefig(save, bbox_inches='tight')
    if show:
        plt.show(fig, block=False)
    return fig, axes

# NOTE: is a bit buggy, maybe reactivate at some later time point
# def est_corner_bounds_plot(estimation, num_iter=14, settings=None, show=True, save=None):
#     """Wrapper to cornerbound plots of `dynesty <https://dynesty.readthedocs.io/en/latest/quickstart.html>`_
#     nested sampling module.
#
#     Parameters
#     ----------
#     estimation : memocell.estimation.Estimation
#         A memocell estimation object.
#     num_iter : int, optional
#         Number of iterations/subplots (iterations that are selected for plotting
#         lie uniformly between first and last iteration of the nested sampling run).
#     settings : dict of dict, optional
#         Optional labels for parameters.
#     show : bool, optional
#         Plot is shown if `show=True`.
#     save : None or str, optional
#         Provide a path to save the plot.
#
#     Returns
#     -------
#     fig : matplotlib.figure.Figure
#     axes : list or array of matplotlib.axes
#     """
#
#     # if not given, create some default settings
#     if settings==None:
#         settings = dict()
#         for theta_id in estimation.net.net_theta_symbolic:
#             param = estimation.net.net_rates_identifier[theta_id]
#             settings[param] = {'label': param}
#
#     # get plotting information from estimation instance
#     sampler_result, params_labels, prior_transform = estimation._sampling_res_and_labels_and_priortransform(settings)
#
#     # create subplots
#     num_it_segs = num_iter
#     num_it_total = sampler_result.niter
#     num_subplot_rows = int(np.ceil(num_it_segs/3.0))
#     fig, axes = plt.subplots(num_subplot_rows, 3, figsize=(num_subplot_rows*3, 3*3))
#
#     # make square
#     for ax in axes.flatten():
#         ax.set(aspect='equal')
#
#     # turn excess subplots off
#     for i in range(num_it_segs, num_subplot_rows*3):
#         axes.flatten()[i].set_axis_off()
#
#     # actual plotting
#     for it_plot in range(num_it_segs):
#         it_num = int(it_plot * num_it_total / float(num_it_segs - 1))
#         it_num = min(num_it_total, it_num)
#
#         dyplot.cornerbound(sampler_result,
#                     it=it_num,
#                     prior_transform=prior_transform,
#                     color='lightgrey',
#                     show_live=True,
#                     live_color='darkorange',
#                     labels=params_labels,
#                     fig=(fig, axes.flatten()[it_plot]))
#
#     plt.tight_layout()
#
#     # save/show figure
#     if save!=None:
#         plt.savefig(save, bbox_inches='tight')
#     if show:
#         plt.show(fig)
#     return fig, axes


def est_chains_plot(estimation, weighted=True,
                    xlabel='Sample iteration', xlim=None, xlog=False,
                    ylabel='Parameter values', ylim=None, ylog=False,
                    show=True, save=None):
    """Plot (weighted) parameter samples over the iterations of the nested sampling
    run.

    Parameters
    ----------
    estimation : memocell.estimation.Estimation
        A memocell estimation object.
    weighted : bool, optional
        Plot parameter samples of the nested sampling run weighted (default)
        or unweighted. Samples need to be weighted to correspond to the actual
        parameter posterior.
    xlabel : str, optional
        Label for x-axis.
    xlim : None or tuple of floats, optional
        Specify x-axis limits.
    xlog : bool, optional
        Logarithmic x-axis if `xlog=True`.
    ylabel : str, optional
        Label for y-axis.
    ylim : None or tuple of floats, optional
        Specify y-axis limits.
    ylog : bool, optional
        Logarithmic y-axis if `ylog=True`.
    show : bool, optional
        Plot is shown if `show=True`.
    save : None or str, optional
        Provide a path to save the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : list or array of matplotlib.axes
    """

    # get plotting information from estimation instance
    if weighted:
        samples, num_params = estimation._samples_weighted_chains_parameters()
    else:
        samples, num_params = estimation._samples_chains_parameters()

    # generate plot
    fig = plt.figure()
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    # set a color cycle for the different params
    colormap = plt.cm.viridis
    plt.gca().set_prop_cycle(cycler('color', [colormap(i) for i in np.linspace(0, 1, num_params)]))

    plt.plot(samples, alpha=0.75)

    # final axis setting
    ax.set_xlim(xlim)
    ax.set_xlabel(xlabel, color="black")
    ax.set_xscale('log' if xlog==True else 'linear')
    ax.set_ylim(ylim)
    ax.set_ylabel(ylabel, color="black")
    ax.set_yscale('log' if ylog==True else 'linear')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # save/show figure
    if save!=None:
        plt.savefig(save, bbox_inches='tight')
    if show:
        plt.show(fig, block=False)
    return fig, fig.axes


def est_bestfit_mean_plot(estimation, settings=None, data=True, cred=True,
                            xlabel='Time', xlim=None, xlog=False,
                            ylabel='Mean', ylim=None, ylog=False,
                            show=True, save=None):
    """Plot the model mean trajectories based on the estimated parameter posterior
    distribution. `Note:` A summary model trajectory is shown based on the median of the individual
    1-dimensional marginal parameter posteriors (`cred=False`) or based on the 50th
    percentile of model trajectory samples from the complete parameter posterior; for multimodal
    parameter posteriors the `cred=False` option can yield an inaccurate summary
    (here, it is advised to use `cred=True`).

    Parameters
    ----------
    estimation : memocell.estimation.Estimation
        A memocell estimation object.
    settings : dict of dict, optional
        Optional label and color settings for the mean traces.
    data : bool, optional
        If `data=True` (default), plot the data mean statistics with standard errors
        in the background (grey color).
    cred : bool, optional
        If `cred=True` (default), a median trajectory with 95% credible bands of the
        model means are shown;
        based on 2.5th, 50th and 97.5th percentiles for each time point of model
        trajectories drawn from the estimated parameter posterior distribution.
        `Note:` The computation of credible bands can be expensive for complex
        models; in this case you might want to start plotting with `cred=False`.
        If `cred=False`, no band is shown and the summary trajectory is computed
        by taking median parameters of the individual 1-dimensional posterior marginals.
    xlabel : str, optional
        Label for x-axis.
    xlim : None or tuple of floats, optional
        Specify x-axis limits.
    xlog : bool, optional
        Logarithmic x-axis if `xlog=True`.
    ylabel : str, optional
        Label for y-axis.
    ylim : None or tuple of floats, optional
        Specify y-axis limits.
    ylog : bool, optional
        Logarithmic y-axis if `ylog=True`.
    show : bool, optional
        Plot is shown if `show=True`.
    save : None or str, optional
        Provide a path to save the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : list or array of matplotlib.axes
    """

    # if not given, create some default settings
    if settings==None:
        vars = [estimation.net_simulation.sim_variables_identifier[var[0]][0] for var in estimation.net_simulation.sim_variables_order[0]]
        colors = sns.color_palette('Set2', n_colors=len(vars)).as_hex()
        settings = dict()

        for i, var in enumerate(vars):
            settings[var] = {'label': f'E({var})', 'color': colors[i]}

    # create plotting information from estimation instance and
    # plot by respective utility plotting function
    if data:
        if cred:
            x_arr_dots, x_arr_line, y_dots_err, y_line, y_lower, y_upper, attributes = estimation._dots_w_bars_and_line_w_band_evolv_mean_credible(settings)

            fig, axes = _dots_w_bars_and_line_w_band_evolv(x_arr_dots, x_arr_line, y_dots_err,
                                            y_line, y_lower, y_upper, attributes,
                                            xlabel=xlabel, xlim=xlim, xlog=xlog,
                                            ylabel=ylabel, ylim=ylim, ylog=ylog,
                                            show=show, save=save)
        else:
            x_arr_dots, x_arr_line, y_dots_err, y_line, attributes = estimation._dots_w_bars_and_line_evolv_bestfit_mean_data(settings)

            fig, axes = _dots_w_bars_and_line_evolv(x_arr_dots, x_arr_line, y_dots_err, y_line, attributes,
                                            xlabel=xlabel, xlim=xlim, xlog=xlog,
                                            ylabel=ylabel, ylim=ylim, ylog=ylog,
                                            show=show, save=save)

    else:
        if cred:
            x_arr, y_line, y_lower, y_upper, attributes = estimation._line_w_band_evolv_mean_credible(settings)

            fig, axes = _line_w_band_evolv(x_arr, y_line, y_lower, y_upper, attributes,
                                            xlabel=xlabel, xlim=xlim, xlog=xlog,
                                            ylabel=ylabel, ylim=ylim, ylog=ylog,
                                            show=show, save=save)
        else:
            x_arr, y_arr, attributes = estimation._line_evolv_bestfit_mean(settings)

            fig, axes = _line_evolv(x_arr, y_arr, attributes,
                                            xlabel=xlabel, xlim=xlim, xlog=xlog,
                                            ylabel=ylabel, ylim=ylim, ylog=ylog,
                                            show=show, save=save)
    return fig, axes

def est_bestfit_variance_plot(estimation, settings=None, data=True, cred=True,
                            xlabel='Time', xlim=None, xlog=False,
                            ylabel='Variance', ylim=None, ylog=False,
                            show=True, save=None):
    """Plot the model variance trajectories based on the estimated parameter posterior
    distribution. `Note:` A summary model trajectory is shown based on the median of the individual
    1-dimensional marginal parameter posteriors (`cred=False`) or based on the 50th
    percentile of model trajectory samples from the complete parameter posterior; for multimodal
    parameter posteriors the `cred=False` option can yield an inaccurate summary
    (here, it is advised to use `cred=True`).

    Parameters
    ----------
    estimation : memocell.estimation.Estimation
        A memocell estimation object.
    settings : dict of dict, optional
        Optional label and color settings for the variance traces.
    data : bool, optional
        If `data=True` (default), plot the data variance statistics with standard errors
        in the background (grey color), if these data are available.
    cred : bool, optional
        If `cred=True` (default), a median trajectory with 95% credible bands of the
        model variances are shown;
        based on 2.5th, 50th and 97.5th percentiles for each time point of model
        trajectories drawn from the estimated parameter posterior distribution.
        `Note:` The computation of credible bands can be expensive for complex
        models; in this case you might want to start plotting with `cred=False`.
        If `cred=False`, no band is shown and the summary trajectory is computed
        by taking median parameters of the individual 1-dimensional posterior marginals.
    xlabel : str, optional
        Label for x-axis.
    xlim : None or tuple of floats, optional
        Specify x-axis limits.
    xlog : bool, optional
        Logarithmic x-axis if `xlog=True`.
    ylabel : str, optional
        Label for y-axis.
    ylim : None or tuple of floats, optional
        Specify y-axis limits.
    ylog : bool, optional
        Logarithmic y-axis if `ylog=True`.
    show : bool, optional
        Plot is shown if `show=True`.
    save : None or str, optional
        Provide a path to save the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : list or array of matplotlib.axes
    """

    # if not given, create some default settings
    if settings==None:
        vars = [(estimation.net_simulation.sim_variables_identifier[var[0]][0], estimation.net_simulation.sim_variables_identifier[var[1]][0])
                    for var in estimation.net_simulation.sim_variables_order[1] if var[0]==var[1]]
        colors = sns.color_palette('Set2', n_colors=len(vars)).as_hex()
        settings = dict()

        for i, var in enumerate(vars):
            settings[var] = {'label': f'Var({var[0]})', 'color': colors[i]}

    # create plotting information from estimation instance and
    # plot by respective utility plotting function
    if data:
        if cred:
            x_arr_dots, x_arr_line, y_dots_err, y_line, y_lower, y_upper, attributes = estimation._dots_w_bars_and_line_w_band_evolv_variance_credible(settings)

            fig, axes = _dots_w_bars_and_line_w_band_evolv(x_arr_dots, x_arr_line, y_dots_err,
                                            y_line, y_lower, y_upper, attributes,
                                            xlabel=xlabel, xlim=xlim, xlog=xlog,
                                            ylabel=ylabel, ylim=ylim, ylog=ylog,
                                            show=show, save=save)
        else:
            x_arr_dots, x_arr_line, y_dots_err, y_line, attributes = estimation._dots_w_bars_and_line_evolv_bestfit_variance_data(settings)

            fig, axes = _dots_w_bars_and_line_evolv(x_arr_dots, x_arr_line, y_dots_err, y_line, attributes,
                                            xlabel=xlabel, xlim=xlim, xlog=xlog,
                                            ylabel=ylabel, ylim=ylim, ylog=ylog,
                                            show=show, save=save)

    else:
        if cred:
            x_arr, y_line, y_lower, y_upper, attributes = estimation._line_w_band_evolv_variance_credible(settings)

            fig, axes = _line_w_band_evolv(x_arr, y_line, y_lower, y_upper, attributes,
                                            xlabel=xlabel, xlim=xlim, xlog=xlog,
                                            ylabel=ylabel, ylim=ylim, ylog=ylog,
                                            show=show, save=save)
        else:
            x_arr, y_arr, attributes = estimation._line_evolv_bestfit_variance(settings)

            fig, axes = _line_evolv(x_arr, y_arr, attributes,
                                            xlabel=xlabel, xlim=xlim, xlog=xlog,
                                            ylabel=ylabel, ylim=ylim, ylog=ylog,
                                            show=show, save=save)
    return fig, axes


def est_bestfit_covariance_plot(estimation, settings=None, data=True, cred=True,
                            xlabel='Time', xlim=None, xlog=False,
                            ylabel='Covariance', ylim=None, ylog=False,
                            show=True, save=None):
    """Plot the model covariance trajectories based on the estimated parameter posterior
    distribution. `Note:` A summary model trajectory is shown based on the median of the individual
    1-dimensional marginal parameter posteriors (`cred=False`) or based on the 50th
    percentile of model trajectory samples from the complete parameter posterior; for multimodal
    parameter posteriors the `cred=False` option can yield an inaccurate summary
    (here, it is advised to use `cred=True`).

    Parameters
    ----------
    estimation : memocell.estimation.Estimation
        A memocell estimation object.
    settings : dict of dict, optional
        Optional label and color settings for the covariance traces.
    data : bool, optional
        If `data=True` (default), plot the data covariance statistics with standard errors
        in the background (grey color), if these data are available.
    cred : bool, optional
        If `cred=True` (default), a median trajectory with 95% credible bands of the
        model covariances are shown;
        based on 2.5th, 50th and 97.5th percentiles for each time point of model
        trajectories drawn from the estimated parameter posterior distribution.
        `Note:` The computation of credible bands can be expensive for complex
        models; in this case you might want to start plotting with `cred=False`.
        If `cred=False`, no band is shown and the summary trajectory is computed
        by taking median parameters of the individual 1-dimensional posterior marginals.
    xlabel : str, optional
        Label for x-axis.
    xlim : None or tuple of floats, optional
        Specify x-axis limits.
    xlog : bool, optional
        Logarithmic x-axis if `xlog=True`.
    ylabel : str, optional
        Label for y-axis.
    ylim : None or tuple of floats, optional
        Specify y-axis limits.
    ylog : bool, optional
        Logarithmic y-axis if `ylog=True`.
    show : bool, optional
        Plot is shown if `show=True`.
    save : None or str, optional
        Provide a path to save the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : list or array of matplotlib.axes
    """

    # if not given, create some default settings
    if settings==None:
        vars = [(estimation.net_simulation.sim_variables_identifier[var[0]][0], estimation.net_simulation.sim_variables_identifier[var[1]][0])
                    for var in estimation.net_simulation.sim_variables_order[1] if var[0]!=var[1]]
        colors = sns.color_palette('Set2', n_colors=len(vars)).as_hex()
        settings = dict()

        for i, var in enumerate(vars):
            settings[var] = {'label': f'Cov({var[0]}, {var[1]})', 'color': colors[i]}

    # create plotting information from estimation instance and
    # plot by respective utility plotting function
    if data:
        if cred:
            x_arr_dots, x_arr_line, y_dots_err, y_line, y_lower, y_upper, attributes = estimation._dots_w_bars_and_line_w_band_evolv_covariance_credible(settings)

            fig, axes = _dots_w_bars_and_line_w_band_evolv(x_arr_dots, x_arr_line, y_dots_err,
                                            y_line, y_lower, y_upper, attributes,
                                            xlabel=xlabel, xlim=xlim, xlog=xlog,
                                            ylabel=ylabel, ylim=ylim, ylog=ylog,
                                            show=show, save=save)
        else:
            x_arr_dots, x_arr_line, y_dots_err, y_line, attributes = estimation._dots_w_bars_and_line_evolv_bestfit_covariance_data(settings)

            fig, axes = _dots_w_bars_and_line_evolv(x_arr_dots, x_arr_line, y_dots_err, y_line, attributes,
                                            xlabel=xlabel, xlim=xlim, xlog=xlog,
                                            ylabel=ylabel, ylim=ylim, ylog=ylog,
                                            show=show, save=save)

    else:
        if cred:
            x_arr, y_line, y_lower, y_upper, attributes = estimation._line_w_band_evolv_covariance_credible(settings)

            fig, axes = _line_w_band_evolv(x_arr, y_line, y_lower, y_upper, attributes,
                                            xlabel=xlabel, xlim=xlim, xlog=xlog,
                                            ylabel=ylabel, ylim=ylim, ylog=ylog,
                                            show=show, save=save)
        else:
            x_arr, y_arr, attributes = estimation._line_evolv_bestfit_covariance(settings)

            fig, axes = _line_evolv(x_arr, y_arr, attributes,
                                            xlabel=xlabel, xlim=xlim, xlog=xlog,
                                            ylabel=ylabel, ylim=ylim, ylog=ylog,
                                            show=show, save=save)
    return fig, axes

##################################
### PLOTTING UTILITY FUNCTIONS ###
##################################

def _dots_w_bars(y_arr_err, x_ticks, attributes, show_errorbar=True,
                xlabel=None, xlim=None, xlog=False,
                ylabel=None, ylim=None, ylog=False,
                show=True, save=None):
    """Private plotting utility function."""

    # OLD DOCS, maybe helpful for later use
    # """
    # Plot dots with error bars (values in y axis, iteration over x axis).
    #
    # parameters
    # ----------
    # y_arr_err
    #     numpy array with shape (# dots, 2); errors are given in the second
    #     dimension
    # attributes
    #     dictionary specifying the keys 'color'
    # x_ticks (set None to ignore)
    #     list of labels for dots which are plotted below x axis.
    # output
    #     dictionary specifying the keys 'output_folder' and 'plot_name'
    #
    #
    # example
    # -------
    # y_arr_err = np.array([
    # [1, 0.2],
    # [2, 0.8],
    # [3, 0.3]
    # ])
    #
    # x_ticks = ['a', 'b', 'c']
    #
    # attributes = {'color': 'dodgerblue'}
    #
    # output = {'output_folder': './test_figures',
    #         'plot_name': 'fig_test_dots_bars'}
    # """

    # initialise figure and axis settings
    fig = plt.figure()

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    # actual plotting
    for dot_ind in range(y_arr_err.shape[0]):
        plt.errorbar(dot_ind + 1, y_arr_err[dot_ind, 0],
                    yerr=np.array([y_arr_err[dot_ind, 1], y_arr_err[dot_ind, 2]]).reshape(2,1) if show_errorbar else None,
                    fmt='o', capsize=4, elinewidth=2.5, markeredgewidth=2.5,
                    markersize=5, markeredgecolor=attributes[dot_ind][1],
                    color=attributes[dot_ind][1], ecolor='lightgrey')

    # final axis setting
    ax.set_xlim(xlim)
    ax.set_xlabel(xlabel, color="black")
    ax.set_xscale('log' if xlog==True else 'linear')
    ax.set_ylim(ylim)
    ax.set_ylabel(ylabel, color="black")
    ax.set_yscale('log' if ylog==True else 'linear')

    if x_ticks != None:
        plt.xticks([i + 1 for i in range(y_arr_err.shape[0])],
                    [x_ticks[i] for i in range(y_arr_err.shape[0])], rotation=55)

    # save/show figure
    if save!=None:
        plt.savefig(save, bbox_inches='tight')
    if show:
        plt.show(fig, block=False)
    return fig, fig.axes


def _dots_w_bars_evolv(x_arr, y_arr_err, var_attributes,
                        xlabel=None, xlim=None, xlog=False,
                        ylabel=None, ylim=None, ylog=False,
                        show=True, save=None):
    """Private plotting utility function."""

    # OLD DOCS, maybe helpful for later use
    # """
    # Plot multiple evolving (e.g. over time) dots with error bars.
    #
    # parameters
    # ----------
    # x_arr
    #     numpy array with shape (# time points, )
    # y_arr_err
    #     numpy array with shape (# variables, # time points, 2); third
    #     dimension includes [value, error value]
    # var_attributes
    #     dictionary specifying the keys 0, 1, ..., # variables - 1
    #     with tuples (string for legend label, plt color code for variable)
    # output
    #     dictionary specifying the keys 'output_folder' and 'plot_name'
    #
    # example
    # -------
    # x_arr = np.linspace(0, 2, num=3, endpoint=True)
    # y_arr_err = np.array([
    #                     [[1, 0.1], [2, 0.2], [3, 0.3]],
    #                     [[2, 0.4], [1, 0.1], [4, 0.5]]
    #                     ])
    #
    # # use var_ind as specified in var_order
    # var_attributes = {1: ('$A_t$ (ON gene)', 'limegreen'),
    #                 0: ('$B_t$ (OFF gene)', 'tomato')}
    # output = {'output_folder': './test_figures',
    #         'plot_name': 'fig_test_dots_time'}
    # """

    # internal setting to switch between paper figures and normal
    normal_mode = True # for paper, set False

    # initialise figure and axis settings
    fig = plt.figure()

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    # actual plotting
    for var_ind in range(y_arr_err.shape[0]):
        var_name = var_attributes[var_ind][0]
        var_color = var_attributes[var_ind][1]

        plt.errorbar(x_arr, y_arr_err[var_ind, :, 0], yerr=y_arr_err[var_ind, :, 1],
                    label=var_name, markeredgecolor=var_color, color=var_color, fmt='o',
                    capsize=4.0 if normal_mode else 1.0, elinewidth=2.5 if normal_mode else 0.4,
                    markeredgewidth=2.5 if normal_mode else 0.5, markersize=4.5 if normal_mode else 1.0)

    # final axis setting
    ax.set_xlim(xlim)
    ax.set_xlabel(xlabel, color="black")
    ax.set_xscale('log' if xlog==True else 'linear')
    ax.set_ylim(ylim)
    ax.set_ylabel(ylabel, color="black")
    ax.set_yscale('log' if ylog==True else 'linear')

    # add legend
    legend = ax.legend(loc=0)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('lightgrey')

    # save/show figure
    if save!=None:
        plt.savefig(save, bbox_inches='tight')
    if show:
        plt.show(fig, block=False)
    return fig, fig.axes


def _line_evolv(x_arr, y_line, var_attributes,
                xlabel=None, xlim=None, xlog=False,
                ylabel=None, ylim=None, ylog=False,
                show=True, save=None):
    """Private plotting utility function."""

    # OLD DOCS, maybe helpful for later use
    # """
    # Plot multiple evolving (e.g. over time) lines.
    #
    # parameters
    # ----------
    # x_arr
    #     numpy array with shape (# time points, )
    # y_line
    #     numpy array setting the multiple lines with shape (# variables, # time points)
    # var_attributes
    #     dictionary specifying the keys 0, 1, ..., # variables - 1
    #     with tuples (string for legend label, plt color code for variable)
    # output
    #     dictionary specifying the keys 'output_folder' and 'plot_name'
    #
    # example
    # -------
    # x_arr = np.linspace(0, 2, num=3, endpoint=True)
    #
    # y_line = np.array([
    # [1, 2, 3],
    # [2, 1, 4]
    # ])
    #
    # var_attributes = {1: ('$A_t$ (ON gene)', 'limegreen'),
    #                 0: ('$B_t$ (OFF gene)', 'tomato')}
    # output = {'output_folder': './test_figures',
    #         'plot_name': 'fig_test_line_evolv'}
    # """

    # initialise figure and axis settings
    fig = plt.figure()

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    # actual plotting
    for var_ind in range(y_line.shape[0]):
        var_name = var_attributes[var_ind][0]
        var_color = var_attributes[var_ind][1]

        plt.plot(x_arr, y_line[var_ind, :], label=var_name,
                        color=var_color, linewidth=2.5, zorder=2000)

    # final axis setting
    ax.set_xlim(xlim)
    ax.set_xlabel(xlabel, color="black")
    ax.set_xscale('log' if xlog==True else 'linear')
    ax.set_ylim(ylim)
    ax.set_ylabel(ylabel, color="black")
    ax.set_yscale('log' if ylog==True else 'linear')

    # add legend
    legend = ax.legend(loc=0)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('lightgrey')

    # save/show figure
    if save!=None:
        plt.savefig(save, bbox_inches='tight')
    if show:
        plt.show(fig, block=False)
    return fig, fig.axes


def _line_w_band_evolv(x_arr, y_line, y_lower, y_upper, var_attributes,
                                    xlabel=None, xlim=None, xlog=False,
                                    ylabel=None, ylim=None, ylog=False,
                                    show=True, save=None):
    """Private plotting utility function."""

    # OLD DOCS, maybe helpful for later use
    # """
    # Plot multiple evolving (e.g. over time) lines with error/prediction bands.
    #
    # parameters
    # ----------
    # x_arr
    #     numpy array with shape (# time points, )
    # y_line
    #     numpy array setting the multiple lines with shape (# variables, # time points)
    # y_lower
    #     numpy array setting the lower bounds of the band
    #     with shape (# variables, # time points)
    # y_upper
    #     numpy array setting the upper bounds of the band
    #     with shape (# variables, # time points)
    # var_attributes
    #     dictionary specifying the keys 0, 1, ..., # variables - 1
    #     with tuples (string for legend label, plt color code for variable)
    # output
    #     dictionary specifying the keys 'output_folder' and 'plot_name'
    #
    # example
    # -------
    # x_arr = np.linspace(0, 2, num=3, endpoint=True)
    #
    # y_line = np.array([
    # [1, 2, 3],
    # [2, 1, 4]
    # ])
    #
    # y_lower = np.array([
    # [0.5, 1, 2],
    # [1.8, 0.8, 3.5]
    # ])
    #
    # y_upper = np.array([
    # [1.4, 2.7, 3.1],
    # [2.1, 1.3, 4.4]
    # ])
    #
    # var_attributes = {1: ('$A_t$ (ON gene)', 'limegreen'),
    #                 0: ('$B_t$ (OFF gene)', 'tomato')}
    # output = {'output_folder': './test_figures',
    #         'plot_name': 'fig_test_line_band'}
    # """

    # initialise figure and axis settings
    fig = plt.figure()

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    # actual plotting
    for var_ind in range(y_line.shape[0]):
        var_name = var_attributes[var_ind][0]
        var_color = var_attributes[var_ind][1]

        ax.fill_between(x_arr, y_lower[var_ind, :], y_upper[var_ind, :],
                        color=var_color, alpha=0.5, linewidth=0.0, zorder=1000)
        plt.plot(x_arr, y_line[var_ind, :], label=var_name,
                        color=var_color, linewidth=2.5, zorder=2000)

    # final axis setting
    ax.set_xlim(xlim)
    ax.set_xlabel(xlabel, color="black")
    ax.set_xscale('log' if xlog==True else 'linear')
    ax.set_ylim(ylim)
    ax.set_ylabel(ylabel, color="black")
    ax.set_yscale('log' if ylog==True else 'linear')

    # add legend
    legend = ax.legend(loc=0)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('lightgrey')

    # save/show figure
    if save!=None:
        plt.savefig(save, bbox_inches='tight')
    if show:
        plt.show(fig, block=False)
    return fig, fig.axes


def _dots_w_bars_and_line_evolv(x_arr_dots, x_arr_line,
                                            y_dots_err, y_line, var_attributes,
                                            xlabel=None, xlim=None, xlog=False,
                                            ylabel=None, ylim=None, ylog=False,
                                            show=True, save=None):
    """Private plotting utility function."""

    # OLD DOCS, maybe helpful for later use
    # """
    # Plot multiple evolving (e.g. over time) lines with error/prediction bands.
    #
    # parameters
    # ----------
    # x_arr_dots
    #     numpy array with shape (# time points, ); used for aligment with
    #     y_dots_err
    # x_arr_line
    #     numpy array with shape (# time points, ); used for aligment with
    #     y_line, y_lower and y_upper
    # y_dots_err
    #     numpy array with shape (# variables, # time points, 2); third
    #     dimension includes [value, error value]
    # y_line
    #     numpy array setting the multiple lines with shape (# variables, # time points)
    # var_attributes
    #     dictionary specifying the keys 0, 1, ..., # variables - 1
    #     with tuples (string for legend label, plt color code for variable)
    # output
    #     dictionary specifying the keys 'output_folder' and 'plot_name'
    #
    # example
    # -------
    # x_arr_dots = np.linspace(0, 1, num=2, endpoint=True)
    # x_arr_line = np.linspace(0, 2, num=3, endpoint=True)
    # y_dots_err = np.array([
    #                     [[1, 0.1], [2, 0.3]],
    #                     [[2, 0.4], [1, 0.5]]
    #                     ])
    # y_line = np.array([
    # [1, 2, 3],
    # [2, 1, 4]
    # ])
    # var_attributes = {1: ('$A_t$ (ON gene)', 'limegreen'),
    #                 0: ('$B_t$ (OFF gene)', 'tomato')}
    # output = {'output_folder': './test_figures',
    #         'plot_name': 'fig_test_line_band_dots_bars'}
    # """

    # initialise figure and axis settings
    fig = plt.figure()

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    # actual plotting
    for var_ind in range(y_line.shape[0]):
        var_name = var_attributes[var_ind][0]
        var_color = var_attributes[var_ind][1]

        # ax.fill_between(x_arr_line, y_lower[var_ind, :], y_upper[var_ind, :],
        #                 color=var_color, alpha=0.5, linewidth=0.0, zorder=1000)

        # # normal version
        # plt.plot(x_arr_line, y_line[var_ind, :], label=var_name,
        #                 color=var_color, linewidth=3, zorder=3000)
        # plt.errorbar(x_arr_dots, y_dots_err[var_ind, :, 0], yerr=y_dots_err[var_ind, :, 1],
        #             fmt='o', capsize=4.0, elinewidth=2.5, #label='data' if var_ind==0 else '',
        #             markeredgewidth=2.5, markersize=4.5, markeredgecolor='lightgrey', color='lightgrey', zorder=2000)
        # poster version
        # plt.plot(x_arr_line, y_line[var_ind, :], label=var_name,
        #                 color=var_color, linewidth=2, zorder=3000)
        # plt.errorbar(x_arr_dots, y_dots_err[var_ind, :, 0], yerr=y_dots_err[var_ind, :, 1],
        #             fmt='o', capsize=2, elinewidth=2, #label='data' if var_ind==0 else '',
        #             markeredgewidth=2, markersize=3, markeredgecolor='lightgrey', color='lightgrey', zorder=2000)
        # paper version
        plt.plot(x_arr_line, y_line[var_ind, :], label=var_name,
                        color=var_color, linewidth=2.5, zorder=3000)
        if y_dots_err.shape[0]>0:
            plt.errorbar(x_arr_dots, y_dots_err[var_ind, :, 0], yerr=y_dots_err[var_ind, :, 1],
                        fmt='o', capsize=4.0, elinewidth=2.5, #label='data' if var_ind==0 else '',
                        markeredgewidth=2.5, markersize=4.5, markeredgecolor='lightgrey', color='lightgrey', zorder=2000)

    # final axis setting
    ax.set_xlim(xlim)
    ax.set_xlabel(xlabel, color="black")
    ax.set_xscale('log' if xlog==True else 'linear')
    ax.set_ylim(ylim)
    ax.set_ylabel(ylabel, color="black")
    ax.set_yscale('log' if ylog==True else 'linear')

    # add legend
    legend = ax.legend(loc=0)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('lightgrey')
    plt.legend(frameon=False)

    # save/show figure
    if save!=None:
        plt.savefig(save, bbox_inches='tight')
    if show:
        plt.show(fig, block=False)
    return fig, fig.axes


def _dots_w_bars_and_line_w_band_evolv(x_arr_dots, x_arr_line, y_dots_err, y_line,
                                        y_lower, y_upper, var_attributes,
                                        xlabel=None, xlim=None, xlog=False,
                                        ylabel=None, ylim=None, ylog=False,
                                        show=True, save=None):
    """Private plotting utility function."""

    # OLD DOCS, maybe helpful for later use
    # """
    # Plot multiple evolving (e.g. over time) lines with error/prediction bands.
    #
    # parameters
    # ----------
    # x_arr_dots
    #     numpy array with shape (# time points, ); used for aligment with
    #     y_dots_err
    # x_arr_line
    #     numpy array with shape (# time points, ); used for aligment with
    #     y_line, y_lower and y_upper
    # y_dots_err
    #     numpy array with shape (# variables, # time points, 2); third
    #     dimension includes [value, error value]
    # y_line
    #     numpy array setting the multiple lines with shape (# variables, # time points)
    # y_lower
    #     numpy array setting the lower bounds of the band
    #     with shape (# variables, # time points)
    # y_upper
    #     numpy array setting the upper bounds of the band
    #     with shape (# variables, # time points)
    # var_attributes
    #     dictionary specifying the keys 0, 1, ..., # variables - 1
    #     with tuples (string for legend label, plt color code for variable)
    # output
    #     dictionary specifying the keys 'output_folder' and 'plot_name'
    #
    # example
    # -------
    # x_arr_dots = np.linspace(0, 1, num=2, endpoint=True)
    # x_arr_line = np.linspace(0, 2, num=3, endpoint=True)
    # y_dots_err = np.array([
    #                     [[1, 0.1], [2, 0.3]],
    #                     [[2, 0.4], [1, 0.5]]
    #                     ])
    # y_line = np.array([
    # [1, 2, 3],
    # [2, 1, 4]
    # ])
    # y_lower = np.array([
    # [0.5, 1, 2],
    # [1.8, 0.8, 3.5]
    # ])
    # y_upper = np.array([
    # [1.4, 2.7, 3.1],
    # [2.1, 1.3, 4.4]
    # ])
    # var_attributes = {1: ('$A_t$ (ON gene)', 'limegreen'),
    #                 0: ('$B_t$ (OFF gene)', 'tomato')}
    # output = {'output_folder': './test_figures',
    #         'plot_name': 'fig_test_line_band_dots_bars'}
    # """

    # internal setting to switch between paper figures and normal
    normal_mode = True # for paper, set False

    # initialise figure and axis settings
    fig = plt.figure()

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    # actual plotting
    for var_ind in range(y_line.shape[0]):
        var_name = var_attributes[var_ind][0]
        var_color = var_attributes[var_ind][1]

        ax.fill_between(x_arr_line, y_lower[var_ind, :], y_upper[var_ind, :],
                        color=var_color, alpha=0.5, linewidth=0.0, zorder=1000)
        plt.plot(x_arr_line, y_line[var_ind, :], label=var_name,
                        color=var_color, linewidth=2.5 if normal_mode else 1.0, zorder=3000)
        if y_dots_err.shape[0]>0:
            plt.errorbar(x_arr_dots, y_dots_err[var_ind, :, 0], yerr=y_dots_err[var_ind, :, 1],
                        fmt='o', capsize=4.0 if normal_mode else 1.0, elinewidth=2.5 if normal_mode else 0.5, #label='data' if var_ind==0 else '',
                        markeredgewidth=2.5 if normal_mode else 0.5, markersize=4.5 if normal_mode else 1.0, markeredgecolor='lightgrey', color='lightgrey', zorder=2000)

    # final axis setting
    ax.set_xlim(xlim)
    ax.set_xlabel(xlabel, color="black")
    ax.set_xscale('log' if xlog==True else 'linear')
    ax.set_ylim(ylim)
    ax.set_ylabel(ylabel, color="black")
    ax.set_yscale('log' if ylog==True else 'linear')

    # add legend
    legend = ax.legend(loc=0)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('lightgrey')

    # save/show figure
    if save!=None:
        plt.savefig(save, bbox_inches='tight')
    if show:
        plt.show(fig, block=False)
    return fig, fig.axes


def _histogram_discrete(bar_arr, bar_attributes, normalised=False,
                        xlabel=None, xlim=None, xlog=False,
                        ylabel=None, ylim=None, ylog=False,
                        show=True, save=None):
    """Private plotting utility function."""

    # OLD DOCS, maybe helpful for later use
    # """
    # Plot a histogram for discrete values.
    #
    # parameters
    # ----------
    # bar_arr
    #     numpy array of discrete values with shape (#realisations, #variables),
    #     histograms are computed over all realisations for each variable
    #
    # bar_attributes
    #     dictionary with keys specifying a general bin 'label' and bin 'color'
    # output
    #     dictionary specifying the keys 'output_folder' and 'plot_name'
    #
    # example
    # -------
    # bar_arr = np.random.poisson(10, size=10).reshape(10, 1)
    #
    # bar_attributes = {0 : {'label': 'some bins', 'color': 'dodgerblue', 'opacity': 1.0}}
    #
    # output = {'output_folder': './test_figures',
    #         'plot_name': 'hist_disc_test'}
    # """

    # initialise figure and axis settings
    fig = plt.figure()

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    # plotting of a histogram
    try:
        bar_min = min(np.amin(bar_arr), xlim[0])
    except:
        bar_min = np.amin(bar_arr)

    try:
        bar_max = max(np.amax(bar_arr), xlim[1])
    except:
        bar_max = np.amax(bar_arr)

    hist_bins = np.linspace(bar_min - 0.5, bar_max + 0.5, num=int(bar_max - bar_min + 2))

    for var_ind in range(bar_arr.shape[1]):
        plt.hist(bar_arr[:, var_ind], bins=hist_bins,
            density=normalised,
            histtype='stepfilled', # step, stepfilled
            linewidth=2.0,
            align='mid', color=bar_attributes[var_ind]['color'],
            label=bar_attributes[var_ind]['label'],
            alpha=bar_attributes[var_ind]['opacity'], zorder=1-var_ind)

    # ax.set_xticks(bins + 0.5)

    # final axis setting
    ax.set_xlim(xlim)
    ax.set_xlabel(xlabel, color="black")
    ax.set_xscale('log' if xlog==True else 'linear')
    ax.set_ylim(ylim)
    ax.set_ylabel(ylabel, color="black")
    ax.set_yscale('log' if ylog==True else 'linear')

    # add legend
    legend = ax.legend(loc=0)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('lightgrey')
    plt.legend(frameon=False)

    # save/show figure
    if save!=None:
        plt.savefig(save, bbox_inches='tight')
    if show:
        plt.show(fig, block=False)
    return fig, fig.axes


def _histogram_discrete_w_line(bar_arr, bar_attributes, line_function, normalised=False,
                            xlabel=None, xlim=None, xlog=False,
                            ylabel=None, ylim=None, ylog=False,
                            show=True, save=None):
    """Private plotting utility function."""

    # OLD DOCS, maybe helpful for later use
    # """
    # Plot a histogram for discrete values with line.
    #
    # parameters
    # ----------
    # bar_arr
    #     numpy array of discrete values with shape (#realisations, #variables),
    #     histograms are computed over all realisations for each variable
    #
    # bar_attributes
    #     dictionary with keys specifying a general bin 'label' and bin 'color'
    # output
    #     dictionary specifying the keys 'output_folder' and 'plot_name'
    #
    # example
    # -------
    # bar_arr = np.random.poisson(10, size=10).reshape(10, 1)
    #
    # bar_attributes = {0 : {'label': 'some bins', 'color': 'dodgerblue', 'opacity': 1.0}}
    #
    # output = {'output_folder': './test_figures',
    #         'plot_name': 'hist_disc_test'}
    # """

    # initialise figure and axis settings
    fig = plt.figure()

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    # plotting of a histogram
    try:
        bar_min = min(np.amin(bar_arr), xlim[0])
    except:
        bar_min = np.amin(bar_arr)

    try:
        bar_max = max(np.amax(bar_arr), xlim[1])
    except:
        bar_max = np.amax(bar_arr)

    x_line_arr = np.linspace(bar_min, bar_max, num=1000, endpoint=True)
    hist_bins = np.linspace(bar_min - 0.5, bar_max + 0.5, num=bar_max - bar_min + 2)

    for var_ind in range(bar_arr.shape[1]):
        plt.hist(bar_arr[:, var_ind], bins=hist_bins,
            density=normalised,
            histtype='stepfilled', # step, stepfilled
            linewidth=2.0,
            align='mid', color=bar_attributes[var_ind]['color'],
            label=bar_attributes[var_ind]['label'],
            alpha=bar_attributes[var_ind]['opacity'])

        x_line, y_line, line_label, line_color = line_function(x_line_arr, bar_arr[:, var_ind])
        plt.plot(x_line, y_line, label=line_label, color=line_color, linewidth=2.5)

    # ax.set_xticks(bins + 0.5)

    # final axis setting
    ax.set_xlim(xlim)
    ax.set_xlabel(xlabel, color="black")
    ax.set_xscale('log' if xlog==True else 'linear')
    ax.set_ylim(ylim)
    ax.set_ylabel(ylabel, color="black")
    ax.set_yscale('log' if ylog==True else 'linear')

    # add legend
    legend = ax.legend(loc=0)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('lightgrey')

    # save/show figure
    if save!=None:
        plt.savefig(save, bbox_inches='tight')
    if show:
        plt.show(fig, block=False)
    return fig, fig.axes


def _histogram_continuous(bar_arr, bar_attributes, normalised=False,
                        xlabel=None, xlim=None, xlog=False,
                        ylabel=None, ylim=None, ylog=False,
                        show=True, save=None):
    """Private plotting utility function."""

    # OLD DOCS, maybe helpful for later use
    # """
    # Plot a histogram for continuous values.
    #
    # parameters
    # ----------
    # bar_arr
    #     numpy array of continuous values with shape (#realisations, #variables),
    #     histograms are computed over all realisations for each variable
    #
    # bar_attributes
    #     dictionary with keys specifying a general bin 'label', bin 'color',
    #     bin 'edges' and bin edges 'interval_type' ('[)' (default) or '(]')
    #
    # normalised
    #     True or False
    #
    # interval
    #
    # output
    #     dictionary specifying the keys 'output_folder' and 'plot_name'
    #
    # example
    # -------
    # bar_arr = np.random.poisson(10, size=10).reshape(10, 1)
    #
    # bar_attributes = {0 : {'label': 'some bins', 'color': 'dodgerblue', 'opacity': 1.0}}
    #
    # output = {'output_folder': './test_figures',
    #         'plot_name': 'hist_disc_test'}
    # """

    # initialise figure and axis settings
    fig = plt.figure()

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    # plotting of a histogram
    for var_ind in range(bar_arr.shape[1]):
        # read out interval type, default plt.hist() behavior are
        # half-open interval [.., ..), small epsilon shift mimicks (.., ..]
        if bar_attributes[var_ind]['interval_type']=='(]':
            epsilon = 1e-06
        else:
            epsilon = 0.0

        plt.hist(bar_arr[:, var_ind] - epsilon,
            bins=bar_attributes[var_ind]['edges'],
            density=normalised,
            histtype='stepfilled', # step, stepfilled
            linewidth=2.0,
            color=bar_attributes[var_ind]['color'],
            label=bar_attributes[var_ind]['label'],
            alpha=bar_attributes[var_ind]['opacity'])

    # final axis setting
    ax.set_xlim(xlim)
    ax.set_xlabel(xlabel, color="black")
    ax.set_xscale('log' if xlog==True else 'linear')
    ax.set_ylim(ylim)
    ax.set_ylabel(ylabel, color="black")
    ax.set_yscale('log' if ylog==True else 'linear')

    # add legend
    legend = ax.legend(loc=0)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('lightgrey')

    # save/show figure
    if save!=None:
        plt.savefig(save, bbox_inches='tight')
    if show:
        plt.show(fig, block=False)
    return fig, fig.axes


def _histogram_continuous_w_line(bar_arr, bar_attributes, line_function, normalised=False,
                                xlabel=None, xlim=None, xlog=False,
                                ylabel=None, ylim=None, ylog=False,
                                show=True, save=None):
    """Private plotting utility function."""

    # OLD DOCS, maybe helpful for later use
    # """
    # Plot a histogram for continuous values with line.
    #
    # parameters
    # ----------
    # bar_arr
    #     numpy array of continuous values with shape (#realisations, #variables),
    #     histograms are computed over all realisations for each variable
    #
    # bar_attributes
    #     dictionary with keys specifying a general bin 'label', bin 'color',
    #     bin 'edges' and bin edges 'interval_type' ('[)' (default) or '(]')
    #
    # normalised
    #     True or False
    #
    # interval
    #
    # output
    #     dictionary specifying the keys 'output_folder' and 'plot_name'
    #
    # example
    # -------
    # bar_arr = np.random.poisson(10, size=10).reshape(10, 1)
    #
    # bar_attributes = {0 : {'label': 'some bins', 'color': 'dodgerblue', 'opacity': 1.0}}
    #
    # output = {'output_folder': './test_figures',
    #         'plot_name': 'hist_disc_test'}
    # """

    # initialise figure and axis settings
    fig = plt.figure()

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    # plotting of a histogram
    for var_ind in range(bar_arr.shape[1]):
        # read out interval type, default plt.hist() behavior are
        # half-open interval [.., ..), small epsilon shift mimicks (.., ..]
        if bar_attributes[var_ind]['interval_type']=='(]':
            epsilon = 1e-06
        else:
            epsilon = 0.0

        plt.hist(bar_arr[:, var_ind] - epsilon,
            bins=bar_attributes[var_ind]['edges'],
            density=normalised,
            histtype='stepfilled', # step, stepfilled
            linewidth=2.0,
            color=bar_attributes[var_ind]['color'],
            label=bar_attributes[var_ind]['label'],
            alpha=bar_attributes[var_ind]['opacity'])

        x_line_arr = np.linspace(np.amin(bar_attributes[var_ind]['edges']),
                                np.amax(bar_attributes[var_ind]['edges']),
                                num=1000, endpoint=True)
        x_line, y_line, line_label, line_color = line_function(x_line_arr, bar_arr[:, var_ind])
        plt.plot(x_line, y_line, label=line_label, color=line_color, linewidth=2.5)

    # final axis setting
    ax.set_xlim(xlim)
    ax.set_xlabel(xlabel, color="black")
    ax.set_xscale('log' if xlog==True else 'linear')
    ax.set_ylim(ylim)
    ax.set_ylabel(ylabel, color="black")
    ax.set_yscale('log' if ylog==True else 'linear')

    # add legend
    legend = ax.legend(loc=0)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('lightgrey')

    # save/show figure
    if save!=None:
        plt.savefig(save, bbox_inches='tight')
    if show:
        plt.show(fig, block=False)
    return fig, fig.axes


def _scatter(x_arr, y_arr, attributes, xlabel=None, xlim=None, xlog=False,
                                    ylabel=None, ylim=None, ylog=False,
                                    show=True, save=None):
    """Private plotting utility function."""

    # initialise figure and axis settings
    fig = plt.figure()

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    # plotting of a histogram
    plt.scatter(x_arr, y_arr,
                color=attributes['color'],
                alpha=attributes['opacity'],
                label=attributes['label'])

    # ax.set_xticks(bins + 0.5)

    # final axis setting
    ax.set_xlim(xlim)
    ax.set_xlabel(xlabel, color="black")
    ax.set_xscale('log' if xlog==True else 'linear')
    ax.set_ylim(ylim)
    ax.set_ylabel(ylabel, color="black")
    ax.set_yscale('log' if ylog==True else 'linear')

    # add legend
    if not attributes['label']==None:
        legend = ax.legend(loc=0)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_edgecolor('lightgrey')

    # save/show figure
    if save!=None:
        plt.savefig(save, bbox_inches='tight')
    if show:
        plt.show(fig, block=False)
    return fig, fig.axes


### class version (for reference) uncommented June 2020
### (might still contain some more code snippets for later use)
### new version above

# class Plots(object):
#     """
#     Class with multiple methods for plotting.
#
#     methods
#     -------
#     network_graph
#     samples_corner
#     samples_chains
#     dots_w_bars_evolv
#     line_evolv
#     line_w_band_evolv
#     dots_w_bars_and_line_evolv
#     dots_w_bars_and_line_w_band_evolv
#     dots_w_bars
#
#     attributes
#     ----------
#     x_axis
#         dictionary with keys specifying x axis 'label', lower and upper axis
#         'limits' and 'log' scale
#     y_axis
#         dictionary with keys specifying y axis 'label', lower and upper axis
#         'limits' and 'log' scale
#     show=False (default)
#         set to True to show plot
#
#     example
#     -------
#     x_axis = {'label': 'time',
#             'limits': (None, None),
#             'log': False}
#     y_axis = {'label': 'counts',
#             'limits': (None, None),
#             'log': False}
#     """
#     def __init__(self, x_axis, y_axis, show=True, save=False):
#         # initialise basic information
#         try:
#             self.xlabel = x_axis['label']
#             self.xlim = x_axis['limits']
#             self.xlog = x_axis['log']
#
#             self.ylabel = y_axis['label']
#             self.ylim = y_axis['limits']
#             self.ylog = y_axis['log']
#         except:
#             self.xlabel = None
#             self.xlim = None
#             self.xlog = None
#
#             self.ylabel = None
#             self.ylim = None
#             self.ylog = None
#
#         self.plot_show = show
#         self.plot_save = save
#
#         # update or set basic figure settings
#         # NOTE: Helvetica Neue has to be installed, otherwise default font is used
#         # plt.rcParams.update({'figure.autolayout': True}) # replaced with , bbox_inches='tight'
#         # plt.rcParams.update({'figure.figsize': (8, 5)})
#         # plt.rcParams.update({'font.size': 14})
#         # plt.rcParams['font.family'] = 'Helvetica Neue' # uncommented 2020
#         # plt.rcParams['font.weight'] = 'medium'
#         # plt.rcParams['mathtext.fontset'] = 'custom' # uncommented 2020
#         # plt.rcParams['mathtext.rm'] = 'Helvetica Neue' # uncommented 2020
#         # plt.rcParams['mathtext.it'] = 'Helvetica Neue:italic' # uncommented 2020
#         # plt.rcParams['mathtext.rm'] = 'Helvetica Neue:medium'
#         # plt.rcParams['mathtext.it'] = 'Helvetica Neue:medium:italic'
#         # plt.rcParams['axes.labelweight'] = 'medium'
#         # plt.rcParams['axes.labelsize'] = 16
#         # plt.rcParams['axes.linewidth'] = 1.2
#
#
#     def network_graph(self, net_graphviz, layout_engine, output):
#         """docstring for ."""
#
#         ### plotting via networkx
#         # fig = plt.figure(figsize=figsize)
#         # ax = fig.gca()
#         # ax.spines['top'].set_visible(False)
#         # ax.spines['right'].set_visible(False)
#         # ax.spines['bottom'].set_visible(False)
#         # ax.spines['left'].set_visible(False)
#         #
#         # pos = nx.nx_agraph.graphviz_layout(net_graphviz, prog=layout_engine)
#         # nx.draw_networkx_nodes(net_graphviz, pos, node_color=node_colors,
#         #                     node_size=node_sizes)
#         #
#         # nx.draw_networkxlabels(net_graphviz, pos,
#         #               labels=node_labels, font_color='black') # , font_size=0.5
#         #
#         # # draw network edges
#         # # (node_size is given otherwise arrowheads are displaced)
#         # nx.draw_networkx_edges(net_graphviz, pos, edge_color=edge_colors,
#         #                 arrowstyle='-|>') #, connectionstyle='arc3,rad=-0.3', arrowsize=6) # , node_size=node_size, alpha=edge_alpha)
#         #
#         # # possible arrow styles that look nice
#         # # '-|>' (default)
#         # # '->'
#         #
#         # nx.draw_networkx_edge_labels(net_graphviz, pos, edge_labels=edge_labels,
#         #                 label_pos=0.4)
#         ###
#
#         pdot = nx.drawing.nx_pydot.to_pydot(net_graphviz)
#
#         # save/show figure
#         if self.plot_save:
#             # plt.savefig(output['output_folder'] + '/{0}.pdf'.format(output['plot_name']), bbox_inches='tight') # networkx
#             pdot.write_pdf(output['output_folder'] + '/{0}.pdf'.format(output['plot_name']), prog=layout_engine)
#
#         if self.plot_show:
#             # plt.show() # networkx
#             display(Image(pdot.create_png(prog=layout_engine)))
#
#         # # graphviz version
#         # A = to_agraph(net_graphviz)
#         # A.layout(layout_engine)
#         # A.draw(output['output_folder'] + '/{0}.pdf'.format(output['plot_name']))
#         #
#         # if self.plot_show:
#         #     graphviz.Source(A).view()
#
#         # NOTE: use this somehow (arrow option for directed graphs in networkx)
#         # nx.draw_networkx(G, arrows=True, **options)
#         # options = {
#         #     'node_color': 'blue',
#         #     'node_size': 100,
#         #     'width': 3,
#         #     'arrowstyle': '-|>',
#         #     'arrowsize': 12,
#         # }
#
#
#     def samples_corner(self, samples, labels, output):
#         """docstring for ."""
#         # initialise figure
#         # plt.rcParams.update({'figure.autolayout': True})
#         # plt.rcParams.update({'font.size': 16})
#         # plt.rcParams['font.family'] = 'Helvetica Neue' # uncommented 2020
#         # plt.rcParams['font.weight'] = 'medium'
#         # plt.rcParams['mathtext.fontset'] = 'custom' # uncommented 2020
#         # plt.rcParams['mathtext.rm'] = 'Helvetica Neue' # uncommented 2020
#         # plt.rcParams['mathtext.it'] = 'Helvetica Neue:italic' # uncommented 2020
#         # plt.rcParams['mathtext.rm'] = 'Helvetica Neue:medium'
#         # plt.rcParams['mathtext.it'] = 'Helvetica Neue:medium:italic'
#
#         # use corner package for this plot
#         fig = corner.corner(samples, labels=labels)
#
#         # save/show figure
#         if self.plot_save:
#             plt.savefig(output['output_folder'] + '/{0}.pdf'.format(output['plot_name']), bbox_inches='tight')
#         if self.plot_show:
#             plt.show(fig, block=False)
#         plt.close(fig)
#         plt.close('all')
#
#
#     def samples_cornerkernel(self, sampler_result, params_labels, output):
#         """docstring for ."""
#
#         # plt.rcParams.update({'figure.autolayout': True})
#         # plt.rcParams.update({'font.size': 16})
#         # plt.rcParams['font.family'] = 'Helvetica Neue' # uncommented 2020
#         # plt.rcParams['font.weight'] = 'medium'
#         # plt.rcParams['mathtext.fontset'] = 'custom' # uncommented 2020
#         # plt.rcParams['mathtext.rm'] = 'Helvetica Neue' # uncommented 2020
#         # plt.rcParams['mathtext.it'] = 'Helvetica Neue:italic' # uncommented 2020
#         # plt.rcParams['mathtext.rm'] = 'Helvetica Neue:medium'
#         # plt.rcParams['mathtext.it'] = 'Helvetica Neue:medium:italic'
#
#         # fig = plt.figure()
#         # ax = fig.gca()
#
#         fig, axes = dyplot.cornerplot(sampler_result,
#                                 color='dodgerblue',
#                                 show_titles=True,
#                                 labels=params_labels,
#                                 title_fmt='.4f')
#         fig.tight_layout()
#
#         # save/show figure
#         if self.plot_save:
#             name = output['plot_name']
#             plt.savefig(output['output_folder'] + f'/{name}.pdf', bbox_inches='tight')
#         if self.plot_show:
#             plt.show(fig, block=False)
#         plt.close(fig)
#         plt.close('all')
#
#
#     def samples_cornerpoints(self, sampler_result, params_labels, output):
#         """docstring for ."""
#
#         # plt.rcParams.update({'figure.autolayout': True})
#         # plt.rcParams.update({'font.size': 16})
#         # plt.rcParams['font.family'] = 'Helvetica Neue' # uncommented 2020
#         # plt.rcParams['font.weight'] = 'medium'
#         # plt.rcParams['mathtext.fontset'] = 'custom' # uncommented 2020
#         # plt.rcParams['mathtext.rm'] = 'Helvetica Neue' # uncommented 2020
#         # plt.rcParams['mathtext.it'] = 'Helvetica Neue:italic' # uncommented 2020
#         # plt.rcParams['mathtext.rm'] = 'Helvetica Neue:medium'
#         # plt.rcParams['mathtext.it'] = 'Helvetica Neue:medium:italic'
#
#         # fig = plt.figure()
#         # ax = fig.gca()
#
#         fig, axes = dyplot.cornerpoints(sampler_result,
#                              cmap='magma',
#                              labels=params_labels)
#
#         # save/show figure
#         if self.plot_save:
#             name = output['plot_name']
#             plt.savefig(output['output_folder'] + f'/{name}.pdf', bbox_inches='tight')
#         if self.plot_show:
#             plt.show(fig, block=False)
#         plt.close(fig)
#         plt.close('all')
#
#
#     def samples_cornerbounds(self, sampler_result, params_labels, prior_transform, output):
#         """docstring for ."""
#
#         # plt.rcParams.update({'figure.autolayout': True})
#         # plt.rcParams.update({'font.size': 16})
#         # plt.rcParams['font.family'] = 'Helvetica Neue' # uncommented 2020
#         # plt.rcParams['font.weight'] = 'medium' # uncommented 2020
#         # plt.rcParams['mathtext.fontset'] = 'custom' # uncommented 2020
#         # plt.rcParams['mathtext.rm'] = 'Helvetica Neue:medium' # uncommented 2020
#         # plt.rcParams['mathtext.it'] = 'Helvetica Neue:medium:italic' # uncommented 2020
#
#         num_it_segs = 14 # num_it_segs+1 plots will be plotted
#         num_it_total = sampler_result.niter
#
#         for it_plot in range(num_it_segs):
#
#             it_num = int( it_plot * num_it_total / float(num_it_segs - 1))
#             it_num = min(num_it_total, it_num)
#
#             # fig = plt.figure()
#             # ax = fig.gca()
#
#             fig, axes = dyplot.cornerbound(sampler_result,
#                                     it=it_num,
#                                     prior_transform=prior_transform,
#                                     color='lightgrey',
#                                     show_live=True,
#                                     live_color='darkorange',
#                                     labels=params_labels)
#
#             # save/show figure
#             if self.plot_save:
#                 name = output['plot_name']
#                 plt.savefig(output['output_folder'] + f'/{name}_it{it_num}.pdf', bbox_inches='tight')
#             if self.plot_show:
#                 plt.show(fig, block=False)
#             plt.close(fig)
#             plt.close('all')
#
#
#     def sampling_runplot(self, sampler_result, output):
#         """docstring for ."""
#
#         # plt.rcParams['axes.titleweight'] = 'medium'
#         # plt.rcParams['axes.titlesize'] = 14
#
#         # fig = plt.figure()
#         # ax = fig.gca()
#         # ax.spines['top'].set_visible(False)
#         # ax.spines['right'].set_visible(False)
#         # ax.spines['bottom'].set_visible(True)
#         # ax.spines['left'].set_visible(True)
#
#         fig, ax = dyplot.runplot(sampler_result,
#                                 color='limegreen') # fig, axes =
#
#         # save/show figure
#         if self.plot_save:
#             name = output['plot_name']
#             plt.savefig(output['output_folder'] + f'/{name}.pdf', bbox_inches='tight')
#         if self.plot_show:
#             plt.show(fig, block=False)
#         plt.close(fig)
#         plt.close('all')
#
#
#     def sampling_traceplot(self, sampler_result, params_labels, output):
#         """docstring for ."""
#
#         # plt.rcParams['axes.titleweight'] = 'medium'
#         # plt.rcParams['axes.titlesize'] = 14
#
#         # fig = plt.figure()
#         # ax = fig.gca()
#         # ax.spines['top'].set_visible(False)
#         # ax.spines['right'].set_visible(False)
#         # ax.spines['bottom'].set_visible(True)
#         # ax.spines['left'].set_visible(True)
#
#         fig, axes = dyplot.traceplot(sampler_result,
#                              show_titles=True,
#                              post_color='dodgerblue',
#                              connect_color='darkorange',
#                              trace_cmap='magma',
#                              connect=True,
#                              connect_highlight=range(5),
#                              labels=params_labels #, title_fmt='.4f'
#                              )
#
#         # save/show figure
#         if self.plot_save:
#             name = output['plot_name']
#             plt.savefig(output['output_folder'] + f'/{name}.pdf', bbox_inches='tight')
#         if self.plot_show:
#             plt.show(fig, block=False)
#         plt.close(fig)
#         plt.close('all')
#
#
#     def samples_chains(self, samples, num_params, output):
#         """docstring for ."""
#
#         # plt.rcParams['axes.titleweight'] = 'medium'
#         # plt.rcParams['axes.titlesize'] = 14
#
#         fig = plt.figure()
#         ax = plt.gca()
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#         ax.spines['bottom'].set_visible(True)
#         ax.spines['left'].set_visible(True)
#
#         # set a color cycle for the different params
#         colormap = plt.cm.viridis
#         plt.gca().set_prop_cycle(cycler('color', [colormap(i) for i in np.linspace(0, 1, num_params)]))
#
#         plt.plot(samples, alpha=0.75)
#
#         # final axis setting
#         ax.set_xlim(self.xlim)
#         ax.set_xlabel(self.xlabel, color="black")
#         ax.set_xscale('log' if self.xlog==True else 'linear')
#         ax.set_ylim(self.ylim)
#         ax.set_ylabel(self.ylabel, color="black")
#         ax.set_yscale('log' if self.ylog==True else 'linear')
#
#         plt.xlabel('sample iteration')
#         plt.ylabel('parameter value')
#
#         # save/show figure
#         if self.plot_save:
#             name = output['plot_name']
#             plt.savefig(output['output_folder'] + f'/{name}.pdf', bbox_inches='tight')
#         if self.plot_show:
#             plt.show(fig, block=False)
#         plt.close(fig)
#         plt.close('all')
#
#
#     # def fig_step_evolv(self, x_arr, y_arr, var_attributes, output):
#     #     """
#     #     Plot an evolving (e.g. over time) step function of multiple variables.
#     #
#     #     parameters
#     #     ----------
#     #     x_arr
#     #         numpy array with shape (# time points, )
#     #     y_arr
#     #         numpy array with shape (# variables, # time points)
#     #     var_attributes
#     #         dictionary specifying the keys 0, 1, ..., # variables - 1
#     #         with tuples (string for legend label, plt color code for variable)
#     #     output
#     #         dictionary specifying the keys 'output_folder' and 'plot_name'
#     #
#     #     example
#     #     -------
#     #     x_arr = np.linspace(0, 10, num=11, endpoint=True)
#     #     y_arr = np.array([
#     #                         [1, 1, 2, 2, 1, 2, 3, 4, 3, 3, 3],
#     #                         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#     #                         ])
#     #     var_attributes = {1: ('$A_t$ (ON gene)', 'limegreen'),
#     #                     0: ('$B_t$ (OFF gene)', 'tomato')}
#     #     output = {'output_folder': './test_figures',
#     #             'plot_name': 'fig_test_step'}
#     #     """
#     #     # initialise figure and axis settings
#     #     plt.figure()
#     #     ax = plt.gca()
#     #     ax.spines['top'].set_visible(False)
#     #     ax.spines['right'].set_visible(False)
#     #     ax.spines['bottom'].set_visible(True)
#     #     ax.spines['left'].set_visible(True)
#     #
#     #     # actual plotting
#     #     for var_ind in range(y_arr.shape[0]):
#     #         var_name = var_attributes[var_ind][0]
#     #         var_color = var_attributes[var_ind][1]
#     #
#     #         plt.step(x_arr, y_arr[var_ind, :], label='{0}'.format(var_name),
#     #                     where='mid', linewidth=2.75, color=var_color, alpha=0.75)
#     #         #         plt.plot(time_arr, var_arr[var_ind, :], marker='o',
#     #         #                     label='{0}'.format(var_name),
#     #         #                     linewidth=2.5, markersize=6, markeredgecolor=var_color,
#     #         #                     color=var_color, alpha=0.75)
#     #
#     #     # final axis setting
#     #     ax.set_xlim(self.xlim)
#     #     ax.set_xlabel(self.xlabel, color="black")
#     #     ax.set_xscale('log' if self.xlog==True else 'linear')
#     #     ax.set_ylim(self.ylim)
#     #     ax.set_ylabel(self.ylabel, color="black")
#     #     ax.set_yscale('log' if self.ylog==True else 'linear')
#     #
#     #     # add legend
#     #     legend = ax.legend(loc=0)
#     #     legend.get_frame().set_facecolor('white')
#     #     legend.get_frame().set_edgecolor('lightgrey')
#     #
#     #     # save figure
#     #     plt.savefig(output['output_folder'] + '/{0}.pdf'.format(output['plot_name']))
#     #     if self.plot_show:
#     #         plt.show()
#     #     plt.close()
#
#     def line_evolv(self, x_arr, y_line, var_attributes,
#                     xlabel=None, xlim=None, xlog=False,
#                     ylabel=None, ylim=None, ylog=False,
#                     show=True, save=None):
#         """
#         Plot multiple evolving (e.g. over time) lines.
#
#         parameters
#         ----------
#         x_arr
#             numpy array with shape (# time points, )
#         y_line
#             numpy array setting the multiple lines with shape (# variables, # time points)
#         var_attributes
#             dictionary specifying the keys 0, 1, ..., # variables - 1
#             with tuples (string for legend label, plt color code for variable)
#         output
#             dictionary specifying the keys 'output_folder' and 'plot_name'
#
#         example
#         -------
#         x_arr = np.linspace(0, 2, num=3, endpoint=True)
#
#         y_line = np.array([
#         [1, 2, 3],
#         [2, 1, 4]
#         ])
#
#         var_attributes = {1: ('$A_t$ (ON gene)', 'limegreen'),
#                         0: ('$B_t$ (OFF gene)', 'tomato')}
#         output = {'output_folder': './test_figures',
#                 'plot_name': 'fig_test_line_evolv'}
#         """
#         # initialise figure and axis settings
#         fig = plt.figure()
#
#         ax = plt.gca()
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#         ax.spines['bottom'].set_visible(True)
#         ax.spines['left'].set_visible(True)
#
#         # actual plotting
#         for var_ind in range(y_line.shape[0]):
#             var_name = var_attributes[var_ind][0]
#             var_color = var_attributes[var_ind][1]
#
#             plt.plot(x_arr, y_line[var_ind, :], label=var_name,
#                             color=var_color, linewidth=2.5, zorder=2000)
#
#         # final axis setting
#         ax.set_xlim(xlim)
#         ax.set_xlabel(xlabel, color="black")
#         ax.set_xscale('log' if xlog==True else 'linear')
#         ax.set_ylim(ylim)
#         ax.set_ylabel(ylabel, color="black")
#         ax.set_yscale('log' if ylog==True else 'linear')
#
#         # add legend
#         legend = ax.legend(loc=0)
#         legend.get_frame().set_facecolor('white')
#         legend.get_frame().set_edgecolor('lightgrey')
#
#         # save/show figure
#         if save!=None:
#             plt.savefig(save, bbox_inches='tight')
#         if show:
#             plt.show(fig, block=False)
#
#
#     def dots_w_bars_evolv(self, x_arr, y_arr_err, var_attributes, output):
#         """
#         Plot multiple evolving (e.g. over time) dots with error bars.
#
#         parameters
#         ----------
#         x_arr
#             numpy array with shape (# time points, )
#         y_arr_err
#             numpy array with shape (# variables, # time points, 2); third
#             dimension includes [value, error value]
#         var_attributes
#             dictionary specifying the keys 0, 1, ..., # variables - 1
#             with tuples (string for legend label, plt color code for variable)
#         output
#             dictionary specifying the keys 'output_folder' and 'plot_name'
#
#         example
#         -------
#         x_arr = np.linspace(0, 2, num=3, endpoint=True)
#         y_arr_err = np.array([
#                             [[1, 0.1], [2, 0.2], [3, 0.3]],
#                             [[2, 0.4], [1, 0.1], [4, 0.5]]
#                             ])
#
#         # use var_ind as specified in var_order
#         var_attributes = {1: ('$A_t$ (ON gene)', 'limegreen'),
#                         0: ('$B_t$ (OFF gene)', 'tomato')}
#         output = {'output_folder': './test_figures',
#                 'plot_name': 'fig_test_dots_time'}
#         """
#         # initialise figure and axis settings
#         fig = plt.figure()
#
#         ax = plt.gca()
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#         ax.spines['bottom'].set_visible(True)
#         ax.spines['left'].set_visible(True)
#
#         # actual plotting
#         for var_ind in range(y_arr_err.shape[0]):
#             var_name = var_attributes[var_ind][0]
#             var_color = var_attributes[var_ind][1]
#
#             plt.errorbar(x_arr, y_arr_err[var_ind, :, 0], yerr=y_arr_err[var_ind, :, 1],
#                         label=var_name, markeredgecolor=var_color, color=var_color, fmt='o',
#                         capsize=4.0, elinewidth=2.5, markeredgewidth=2.5, markersize=4.5)
#
#         # final axis setting
#         ax.set_xlim(self.xlim)
#         ax.set_xlabel(self.xlabel, color="black")
#         ax.set_xscale('log' if self.xlog==True else 'linear')
#         ax.set_ylim(self.ylim)
#         ax.set_ylabel(self.ylabel, color="black")
#         ax.set_yscale('log' if self.ylog==True else 'linear')
#
#         # add legend
#         legend = ax.legend(loc=0)
#         legend.get_frame().set_facecolor('white')
#         legend.get_frame().set_edgecolor('lightgrey')
#
#         # save/show figure
#         if self.plot_save:
#             plt.savefig(output['output_folder'] + '/{0}.pdf'.format(output['plot_name']), bbox_inches='tight')
#
#         if self.plot_show:
#             plt.show(fig, block=False)
#         plt.close(fig)
#         plt.close('all')
#
#
#     def line_w_band_evolv(self, x_arr, y_line, y_lower, y_upper, var_attributes, output):
#         """
#         Plot multiple evolving (e.g. over time) lines with error/prediction bands.
#
#         parameters
#         ----------
#         x_arr
#             numpy array with shape (# time points, )
#         y_line
#             numpy array setting the multiple lines with shape (# variables, # time points)
#         y_lower
#             numpy array setting the lower bounds of the band
#             with shape (# variables, # time points)
#         y_upper
#             numpy array setting the upper bounds of the band
#             with shape (# variables, # time points)
#         var_attributes
#             dictionary specifying the keys 0, 1, ..., # variables - 1
#             with tuples (string for legend label, plt color code for variable)
#         output
#             dictionary specifying the keys 'output_folder' and 'plot_name'
#
#         example
#         -------
#         x_arr = np.linspace(0, 2, num=3, endpoint=True)
#
#         y_line = np.array([
#         [1, 2, 3],
#         [2, 1, 4]
#         ])
#
#         y_lower = np.array([
#         [0.5, 1, 2],
#         [1.8, 0.8, 3.5]
#         ])
#
#         y_upper = np.array([
#         [1.4, 2.7, 3.1],
#         [2.1, 1.3, 4.4]
#         ])
#
#         var_attributes = {1: ('$A_t$ (ON gene)', 'limegreen'),
#                         0: ('$B_t$ (OFF gene)', 'tomato')}
#         output = {'output_folder': './test_figures',
#                 'plot_name': 'fig_test_line_band'}
#         """
#         # initialise figure and axis settings
#         fig = plt.figure()
#
#         ax = plt.gca()
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#         ax.spines['bottom'].set_visible(True)
#         ax.spines['left'].set_visible(True)
#
#         # actual plotting
#         for var_ind in range(y_line.shape[0]):
#             var_name = var_attributes[var_ind][0]
#             var_color = var_attributes[var_ind][1]
#
#             ax.fill_between(x_arr, y_lower[var_ind, :], y_upper[var_ind, :],
#                             color=var_color, alpha=0.5, linewidth=0.0, zorder=1000)
#             plt.plot(x_arr, y_line[var_ind, :], label=var_name,
#                             color=var_color, linewidth=2.5, zorder=2000)
#
#         # final axis setting
#         ax.set_xlim(self.xlim)
#         ax.set_xlabel(self.xlabel, color="black")
#         ax.set_xscale('log' if self.xlog==True else 'linear')
#         ax.set_ylim(self.ylim)
#         ax.set_ylabel(self.ylabel, color="black")
#         ax.set_yscale('log' if self.ylog==True else 'linear')
#
#         # add legend
#         legend = ax.legend(loc=0)
#         legend.get_frame().set_facecolor('white')
#         legend.get_frame().set_edgecolor('lightgrey')
#
#         # save/show figure
#         if self.plot_save:
#             plt.savefig(output['output_folder'] + '/{0}.pdf'.format(output['plot_name']), bbox_inches='tight')
#         if self.plot_show:
#             plt.show(fig, block=False)
#         plt.close(fig)
#         plt.close('all')
#
#
#     def dots_w_bars_and_line_evolv(self, x_arr_dots, x_arr_line, y_dots_err, y_line, var_attributes, output):
#         """
#         Plot multiple evolving (e.g. over time) lines with error/prediction bands.
#
#         parameters
#         ----------
#         x_arr_dots
#             numpy array with shape (# time points, ); used for aligment with
#             y_dots_err
#         x_arr_line
#             numpy array with shape (# time points, ); used for aligment with
#             y_line, y_lower and y_upper
#         y_dots_err
#             numpy array with shape (# variables, # time points, 2); third
#             dimension includes [value, error value]
#         y_line
#             numpy array setting the multiple lines with shape (# variables, # time points)
#         var_attributes
#             dictionary specifying the keys 0, 1, ..., # variables - 1
#             with tuples (string for legend label, plt color code for variable)
#         output
#             dictionary specifying the keys 'output_folder' and 'plot_name'
#
#         example
#         -------
#         x_arr_dots = np.linspace(0, 1, num=2, endpoint=True)
#         x_arr_line = np.linspace(0, 2, num=3, endpoint=True)
#         y_dots_err = np.array([
#                             [[1, 0.1], [2, 0.3]],
#                             [[2, 0.4], [1, 0.5]]
#                             ])
#         y_line = np.array([
#         [1, 2, 3],
#         [2, 1, 4]
#         ])
#         var_attributes = {1: ('$A_t$ (ON gene)', 'limegreen'),
#                         0: ('$B_t$ (OFF gene)', 'tomato')}
#         output = {'output_folder': './test_figures',
#                 'plot_name': 'fig_test_line_band_dots_bars'}
#         """
#         # initialise figure and axis settings
#         fig = plt.figure()
#
#         ax = plt.gca()
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#         ax.spines['bottom'].set_visible(True)
#         ax.spines['left'].set_visible(True)
#
#         # actual plotting
#         for var_ind in range(y_line.shape[0]):
#             var_name = var_attributes[var_ind][0]
#             var_color = var_attributes[var_ind][1]
#
#             # ax.fill_between(x_arr_line, y_lower[var_ind, :], y_upper[var_ind, :],
#             #                 color=var_color, alpha=0.5, linewidth=0.0, zorder=1000)
#
#             # # normal version
#             # plt.plot(x_arr_line, y_line[var_ind, :], label=var_name,
#             #                 color=var_color, linewidth=3, zorder=3000)
#             # plt.errorbar(x_arr_dots, y_dots_err[var_ind, :, 0], yerr=y_dots_err[var_ind, :, 1],
#             #             fmt='o', capsize=4.0, elinewidth=2.5, #label='data' if var_ind==0 else '',
#             #             markeredgewidth=2.5, markersize=4.5, markeredgecolor='lightgrey', color='lightgrey', zorder=2000)
#             # poster version
#             # plt.plot(x_arr_line, y_line[var_ind, :], label=var_name,
#             #                 color=var_color, linewidth=2, zorder=3000)
#             # plt.errorbar(x_arr_dots, y_dots_err[var_ind, :, 0], yerr=y_dots_err[var_ind, :, 1],
#             #             fmt='o', capsize=2, elinewidth=2, #label='data' if var_ind==0 else '',
#             #             markeredgewidth=2, markersize=3, markeredgecolor='lightgrey', color='lightgrey', zorder=2000)
#             # paper version
#             plt.plot(x_arr_line, y_line[var_ind, :], label=var_name,
#                             color=var_color, linewidth=2.5, zorder=3000)
#             plt.errorbar(x_arr_dots, y_dots_err[var_ind, :, 0], yerr=y_dots_err[var_ind, :, 1],
#                         fmt='o', capsize=4.0, elinewidth=2.5, #label='data' if var_ind==0 else '',
#                         markeredgewidth=2.5, markersize=4.5, markeredgecolor='lightgrey', color='lightgrey', zorder=2000)
#
#         # final axis setting
#         ax.set_xlim(self.xlim)
#         ax.set_xlabel(self.xlabel, color="black")
#         ax.set_xscale('log' if self.xlog==True else 'linear')
#         ax.set_ylim(self.ylim)
#         ax.set_ylabel(self.ylabel, color="black")
#         ax.set_yscale('log' if self.ylog==True else 'linear')
#
#         # add legend
#         legend = ax.legend(loc=0)
#         legend.get_frame().set_facecolor('white')
#         legend.get_frame().set_edgecolor('lightgrey')
#         plt.legend(frameon=False)
#
#         # save/show figure
#         if self.plot_save:
#             plt.savefig(output['output_folder'] + '/{0}.pdf'.format(output['plot_name']), bbox_inches='tight')
#         if self.plot_show:
#             plt.show(fig, block=False)
#         plt.close(fig)
#         plt.close('all')
#
#
#     def dots_w_bars_and_line_w_band_evolv(self, x_arr_dots, x_arr_line, y_dots_err, y_line, y_lower, y_upper, var_attributes, output):
#         """
#         Plot multiple evolving (e.g. over time) lines with error/prediction bands.
#
#         parameters
#         ----------
#         x_arr_dots
#             numpy array with shape (# time points, ); used for aligment with
#             y_dots_err
#         x_arr_line
#             numpy array with shape (# time points, ); used for aligment with
#             y_line, y_lower and y_upper
#         y_dots_err
#             numpy array with shape (# variables, # time points, 2); third
#             dimension includes [value, error value]
#         y_line
#             numpy array setting the multiple lines with shape (# variables, # time points)
#         y_lower
#             numpy array setting the lower bounds of the band
#             with shape (# variables, # time points)
#         y_upper
#             numpy array setting the upper bounds of the band
#             with shape (# variables, # time points)
#         var_attributes
#             dictionary specifying the keys 0, 1, ..., # variables - 1
#             with tuples (string for legend label, plt color code for variable)
#         output
#             dictionary specifying the keys 'output_folder' and 'plot_name'
#
#         example
#         -------
#         x_arr_dots = np.linspace(0, 1, num=2, endpoint=True)
#         x_arr_line = np.linspace(0, 2, num=3, endpoint=True)
#         y_dots_err = np.array([
#                             [[1, 0.1], [2, 0.3]],
#                             [[2, 0.4], [1, 0.5]]
#                             ])
#         y_line = np.array([
#         [1, 2, 3],
#         [2, 1, 4]
#         ])
#         y_lower = np.array([
#         [0.5, 1, 2],
#         [1.8, 0.8, 3.5]
#         ])
#         y_upper = np.array([
#         [1.4, 2.7, 3.1],
#         [2.1, 1.3, 4.4]
#         ])
#         var_attributes = {1: ('$A_t$ (ON gene)', 'limegreen'),
#                         0: ('$B_t$ (OFF gene)', 'tomato')}
#         output = {'output_folder': './test_figures',
#                 'plot_name': 'fig_test_line_band_dots_bars'}
#         """
#         # initialise figure and axis settings
#         fig = plt.figure()
#
#         ax = plt.gca()
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#         ax.spines['bottom'].set_visible(True)
#         ax.spines['left'].set_visible(True)
#
#         # actual plotting
#         for var_ind in range(y_line.shape[0]):
#             var_name = var_attributes[var_ind][0]
#             var_color = var_attributes[var_ind][1]
#
#             ax.fill_between(x_arr_line, y_lower[var_ind, :], y_upper[var_ind, :],
#                             color=var_color, alpha=0.5, linewidth=0.0, zorder=1000)
#             plt.plot(x_arr_line, y_line[var_ind, :], label=var_name,
#                             color=var_color, linewidth=2.5, zorder=3000)
#             plt.errorbar(x_arr_dots, y_dots_err[var_ind, :, 0], yerr=y_dots_err[var_ind, :, 1],
#                         fmt='o', capsize=4.0, elinewidth=2.5, #label='data' if var_ind==0 else '',
#                         markeredgewidth=2.5, markersize=4.5, markeredgecolor='lightgrey', color='lightgrey', zorder=2000)
#
#         # final axis setting
#         ax.set_xlim(self.xlim)
#         ax.set_xlabel(self.xlabel, color="black")
#         ax.set_xscale('log' if self.xlog==True else 'linear')
#         ax.set_ylim(self.ylim)
#         ax.set_ylabel(self.ylabel, color="black")
#         ax.set_yscale('log' if self.ylog==True else 'linear')
#
#         # add legend
#         legend = ax.legend(loc=0)
#         legend.get_frame().set_facecolor('white')
#         legend.get_frame().set_edgecolor('lightgrey')
#
#         # save/show figure
#         if self.plot_save:
#             plt.savefig(output['output_folder'] + '/{0}.pdf'.format(output['plot_name']), bbox_inches='tight')
#         if self.plot_show:
#             plt.show(fig, block=False)
#         plt.close(fig)
#         plt.close('all')
#
#
#     # # TODO: steady state plots / volume plots?
#     # def fig_dots_w_bars(self, y_arr_err, attributes, output, x_ticks=None):
#     #     """
#     #     Plot dots with error bars (values in y axis, iteration over x axis).
#     #
#     #     parameters
#     #     ----------
#     #     y_arr_err
#     #         numpy array with shape (# dots, 2); errors are given in the second
#     #         dimension
#     #     attributes
#     #         dictionary specifying the keys 'color'
#     #     output
#     #         dictionary specifying the keys 'output_folder' and 'plot_name'
#     #     x_ticks (optional)
#     #         list of labels for dots which are plotted below x axis.
#     #
#     #     example
#     #     -------
#     #     y_arr_err = np.array([
#     #     [1, 0.2],
#     #     [2, 0.8],
#     #     [3, 0.3]
#     #     ])
#     #
#     #     x_ticks = ['a', 'b', 'c']
#     #
#     #     attributes = {'color': 'dodgerblue'}
#     #
#     #     output = {'output_folder': './test_figures',
#     #             'plot_name': 'fig_test_dots_bars'}
#     #     """
#     #     # initialise figure and axis settings
#     #     plt.figure()
#     #
#     #     ax = plt.gca()
#     #     ax.spines['top'].set_visible(False)
#     #     ax.spines['right'].set_visible(False)
#     #     ax.spines['bottom'].set_visible(True)
#     #     ax.spines['left'].set_visible(True)
#     #
#     #     # actual plotting
#     #     dot_color = attributes['color']
#     #     for dot_ind in range(y_arr_err.shape[0]):
#     #         plt.errorbar(dot_ind + 1, y_arr_err[dot_ind, 0], yerr=y_arr_err[dot_ind, 1],
#     #                     fmt='o', capsize=4, elinewidth=2.5, markeredgewidth=2.5,
#     #                     markersize=5, markeredgecolor=dot_color, color=dot_color, ecolor='lightgrey')
#     #
#     #     # final axis setting
#     #     ax.set_xlim(self.xlim)
#     #     ax.set_xlabel(self.xlabel, color="black")
#     #     ax.set_xscale('log' if self.xlog==True else 'linear')
#     #     ax.set_ylim(self.ylim)
#     #     ax.set_ylabel(self.ylabel, color="black")
#     #     ax.set_yscale('log' if self.ylog==True else 'linear')
#     #
#     #     if x_ticks != None:
#     #         plt.xticks([i + 1 for i in range(y_arr_err.shape[0])],
#     #                     [x_ticks[i] for i in range(y_arr_err.shape[0])], rotation=55)
#     #
#     #     # save figure
#     #     plt.savefig(output['output_folder'] + '/{0}.pdf'.format(output['plot_name']))
#     #     if self.plot_show:
#     #         plt.show()
#     #     plt.close()
#
#
#     def dots_w_bars(self, y_arr_err, x_ticks, attributes, output, show_errorbar=True):
#         """
#         Plot dots with error bars (values in y axis, iteration over x axis).
#
#         parameters
#         ----------
#         y_arr_err
#             numpy array with shape (# dots, 2); errors are given in the second
#             dimension
#         attributes
#             dictionary specifying the keys 'color'
#         x_ticks (set None to ignore)
#             list of labels for dots which are plotted below x axis.
#         output
#             dictionary specifying the keys 'output_folder' and 'plot_name'
#
#
#         example
#         -------
#         y_arr_err = np.array([
#         [1, 0.2],
#         [2, 0.8],
#         [3, 0.3]
#         ])
#
#         x_ticks = ['a', 'b', 'c']
#
#         attributes = {'color': 'dodgerblue'}
#
#         output = {'output_folder': './test_figures',
#                 'plot_name': 'fig_test_dots_bars'}
#         """
#
#         # initialise figure and axis settings
#         fig = plt.figure()
#
#         ax = plt.gca()
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#         ax.spines['bottom'].set_visible(True)
#         ax.spines['left'].set_visible(True)
#
#         # actual plotting
#         for dot_ind in range(y_arr_err.shape[0]):
#             plt.errorbar(dot_ind + 1, y_arr_err[dot_ind, 0],
#                         yerr=np.array([y_arr_err[dot_ind, 1], y_arr_err[dot_ind, 2]]).reshape(2,1) if show_errorbar else None,
#                         fmt='o', capsize=4, elinewidth=2.5, markeredgewidth=2.5,
#                         markersize=5, markeredgecolor=attributes[dot_ind][1],
#                         color=attributes[dot_ind][1], ecolor='lightgrey')
#
#         # final axis setting
#         ax.set_xlim(self.xlim)
#         ax.set_xlabel(self.xlabel, color="black")
#         ax.set_xscale('log' if self.xlog==True else 'linear')
#         ax.set_ylim(self.ylim)
#         ax.set_ylabel(self.ylabel, color="black")
#         ax.set_yscale('log' if self.ylog==True else 'linear')
#
#         if x_ticks != None:
#             plt.xticks([i + 1 for i in range(y_arr_err.shape[0])],
#                         [x_ticks[i] for i in range(y_arr_err.shape[0])], rotation=55)
#
#         # save/show figure
#         if self.plot_save:
#             plt.savefig(output['output_folder'] + '/{0}.pdf'.format(output['plot_name']), bbox_inches='tight')
#         if self.plot_show:
#             plt.show(fig, block=False)
#         plt.close(fig)
#         plt.close('all')
#
#
#     def histogram_discrete(self, bar_arr, bar_attributes, output, normalised=False):
#         """
#         Plot a histogram for discrete values.
#
#         parameters
#         ----------
#         bar_arr
#             numpy array of discrete values with shape (#realisations, #variables),
#             histograms are computed over all realisations for each variable
#
#         bar_attributes
#             dictionary with keys specifying a general bin 'label' and bin 'color'
#         output
#             dictionary specifying the keys 'output_folder' and 'plot_name'
#
#         example
#         -------
#         bar_arr = np.random.poisson(10, size=10).reshape(10, 1)
#
#         bar_attributes = {0 : {'label': 'some bins', 'color': 'dodgerblue', 'opacity': 1.0}}
#
#         output = {'output_folder': './test_figures',
#                 'plot_name': 'hist_disc_test'}
#         """
#
#         # initialise figure and axis settings
#         fig = plt.figure()
#
#         ax = plt.gca()
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#         ax.spines['bottom'].set_visible(True)
#         ax.spines['left'].set_visible(True)
#
#         # plotting of a histogram
#         try:
#             bar_min = min(np.amin(bar_arr), self.xlim[0])
#         except:
#             bar_min = np.amin(bar_arr)
#
#         try:
#             bar_max = max(np.amax(bar_arr), self.xlim[1])
#         except:
#             bar_max = np.amax(bar_arr)
#
#         hist_bins = np.linspace(bar_min - 0.5, bar_max + 0.5, num=bar_max - bar_min + 2)
#
#         for var_ind in range(bar_arr.shape[1]):
#             plt.hist(bar_arr[:, var_ind], bins=hist_bins,
#                 density=normalised,
#                 histtype='stepfilled', # step, stepfilled
#                 linewidth=2.0,
#                 align='mid', color=bar_attributes[var_ind]['color'],
#                 label=bar_attributes[var_ind]['label'],
#                 alpha=bar_attributes[var_ind]['opacity'], zorder=1-var_ind)
#
#         # ax.set_xticks(bins + 0.5)
#
#         # final axis setting
#         ax.set_xlim(self.xlim)
#         ax.set_xlabel(self.xlabel, color="black")
#         ax.set_xscale('log' if self.xlog==True else 'linear')
#         ax.set_ylim(self.ylim)
#         ax.set_ylabel(self.ylabel, color="black")
#         ax.set_yscale('log' if self.ylog==True else 'linear')
#
#         # add legend
#         legend = ax.legend(loc=0)
#         legend.get_frame().set_facecolor('white')
#         legend.get_frame().set_edgecolor('lightgrey')
#         plt.legend(frameon=False)
#
#         # save/show figure
#         if self.plot_save:
#             plt.savefig(output['output_folder'] + '/{0}.pdf'.format(output['plot_name']), bbox_inches='tight')
#         if self.plot_show:
#             plt.show(fig, block=False)
#         plt.close(fig)
#         plt.close('all')
#
#
#     def histogram_discrete_w_line(self, bar_arr, bar_attributes, line_function, output, normalised=False):
#         """
#         Plot a histogram for discrete values with line.
#
#         parameters
#         ----------
#         bar_arr
#             numpy array of discrete values with shape (#realisations, #variables),
#             histograms are computed over all realisations for each variable
#
#         bar_attributes
#             dictionary with keys specifying a general bin 'label' and bin 'color'
#         output
#             dictionary specifying the keys 'output_folder' and 'plot_name'
#
#         example
#         -------
#         bar_arr = np.random.poisson(10, size=10).reshape(10, 1)
#
#         bar_attributes = {0 : {'label': 'some bins', 'color': 'dodgerblue', 'opacity': 1.0}}
#
#         output = {'output_folder': './test_figures',
#                 'plot_name': 'hist_disc_test'}
#         """
#
#         # initialise figure and axis settings
#         fig = plt.figure()
#
#         ax = plt.gca()
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#         ax.spines['bottom'].set_visible(True)
#         ax.spines['left'].set_visible(True)
#
#         # plotting of a histogram
#         try:
#             bar_min = min(np.amin(bar_arr), self.xlim[0])
#         except:
#             bar_min = np.amin(bar_arr)
#
#         try:
#             bar_max = max(np.amax(bar_arr), self.xlim[1])
#         except:
#             bar_max = np.amax(bar_arr)
#
#         x_line_arr = np.linspace(bar_min, bar_max, num=1000, endpoint=True)
#         hist_bins = np.linspace(bar_min - 0.5, bar_max + 0.5, num=bar_max - bar_min + 2)
#
#         for var_ind in range(bar_arr.shape[1]):
#             plt.hist(bar_arr[:, var_ind], bins=hist_bins,
#                 density=normalised,
#                 histtype='stepfilled', # step, stepfilled
#                 linewidth=2.0,
#                 align='mid', color=bar_attributes[var_ind]['color'],
#                 label=bar_attributes[var_ind]['label'],
#                 alpha=bar_attributes[var_ind]['opacity'])
#
#             x_line, y_line, line_label, line_color = line_function(x_line_arr, bar_arr[:, var_ind])
#             plt.plot(x_line, y_line, label=line_label, color=line_color, linewidth=2.5)
#
#         # ax.set_xticks(bins + 0.5)
#
#         # final axis setting
#         ax.set_xlim(self.xlim)
#         ax.set_xlabel(self.xlabel, color="black")
#         ax.set_xscale('log' if self.xlog==True else 'linear')
#         ax.set_ylim(self.ylim)
#         ax.set_ylabel(self.ylabel, color="black")
#         ax.set_yscale('log' if self.ylog==True else 'linear')
#
#         # add legend
#         legend = ax.legend(loc=0)
#         legend.get_frame().set_facecolor('white')
#         legend.get_frame().set_edgecolor('lightgrey')
#
#         # save/show figure
#         if self.plot_save:
#             plt.savefig(output['output_folder'] + '/{0}.pdf'.format(output['plot_name']), bbox_inches='tight')
#         if self.plot_show:
#             plt.show(fig, block=False)
#         plt.close(fig)
#         plt.close('all')
#
#
#     def histogram_continuous(self, bar_arr, bar_attributes, output, normalised=False):
#         """
#         Plot a histogram for continuous values.
#
#         parameters
#         ----------
#         bar_arr
#             numpy array of continuous values with shape (#realisations, #variables),
#             histograms are computed over all realisations for each variable
#
#         bar_attributes
#             dictionary with keys specifying a general bin 'label', bin 'color',
#             bin 'edges' and bin edges 'interval_type' ('[)' (default) or '(]')
#
#         normalised
#             True or False
#
#         interval
#
#         output
#             dictionary specifying the keys 'output_folder' and 'plot_name'
#
#         example
#         -------
#         bar_arr = np.random.poisson(10, size=10).reshape(10, 1)
#
#         bar_attributes = {0 : {'label': 'some bins', 'color': 'dodgerblue', 'opacity': 1.0}}
#
#         output = {'output_folder': './test_figures',
#                 'plot_name': 'hist_disc_test'}
#         """
#
#         # initialise figure and axis settings
#         fig = plt.figure()
#
#         ax = plt.gca()
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#         ax.spines['bottom'].set_visible(True)
#         ax.spines['left'].set_visible(True)
#
#         # plotting of a histogram
#         for var_ind in range(bar_arr.shape[1]):
#             # read out interval type, default plt.hist() behavior are
#             # half-open interval [.., ..), small epsilon shift mimicks (.., ..]
#             if bar_attributes[var_ind]['interval_type']=='(]':
#                 epsilon = 1e-06
#             else:
#                 epsilon = 0.0
#
#             plt.hist(bar_arr[:, var_ind] - epsilon,
#                 bins=bar_attributes[var_ind]['edges'],
#                 density=normalised,
#                 histtype='stepfilled', # step, stepfilled
#                 linewidth=2.0,
#                 color=bar_attributes[var_ind]['color'],
#                 label=bar_attributes[var_ind]['label'],
#                 alpha=bar_attributes[var_ind]['opacity'])
#
#         # final axis setting
#         ax.set_xlim(self.xlim)
#         ax.set_xlabel(self.xlabel, color="black")
#         ax.set_xscale('log' if self.xlog==True else 'linear')
#         ax.set_ylim(self.ylim)
#         ax.set_ylabel(self.ylabel, color="black")
#         ax.set_yscale('log' if self.ylog==True else 'linear')
#
#         # add legend
#         legend = ax.legend(loc=0)
#         legend.get_frame().set_facecolor('white')
#         legend.get_frame().set_edgecolor('lightgrey')
#
#         # save/show figure
#         if self.plot_save:
#             plt.savefig(output['output_folder'] + '/{0}.pdf'.format(output['plot_name']), bbox_inches='tight')
#         if self.plot_show:
#             plt.show(fig, block=False)
#         plt.close(fig)
#         plt.close('all')
#
#
#     def histogram_continuous_w_line(self, bar_arr, bar_attributes, line_function, output, normalised=False):
#         """
#         Plot a histogram for continuous values with line.
#
#         parameters
#         ----------
#         bar_arr
#             numpy array of continuous values with shape (#realisations, #variables),
#             histograms are computed over all realisations for each variable
#
#         bar_attributes
#             dictionary with keys specifying a general bin 'label', bin 'color',
#             bin 'edges' and bin edges 'interval_type' ('[)' (default) or '(]')
#
#         normalised
#             True or False
#
#         interval
#
#         output
#             dictionary specifying the keys 'output_folder' and 'plot_name'
#
#         example
#         -------
#         bar_arr = np.random.poisson(10, size=10).reshape(10, 1)
#
#         bar_attributes = {0 : {'label': 'some bins', 'color': 'dodgerblue', 'opacity': 1.0}}
#
#         output = {'output_folder': './test_figures',
#                 'plot_name': 'hist_disc_test'}
#         """
#
#         # initialise figure and axis settings
#         fig = plt.figure()
#
#         ax = plt.gca()
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#         ax.spines['bottom'].set_visible(True)
#         ax.spines['left'].set_visible(True)
#
#         # plotting of a histogram
#         for var_ind in range(bar_arr.shape[1]):
#             # read out interval type, default plt.hist() behavior are
#             # half-open interval [.., ..), small epsilon shift mimicks (.., ..]
#             if bar_attributes[var_ind]['interval_type']=='(]':
#                 epsilon = 1e-06
#             else:
#                 epsilon = 0.0
#
#             plt.hist(bar_arr[:, var_ind] - epsilon,
#                 bins=bar_attributes[var_ind]['edges'],
#                 density=normalised,
#                 histtype='stepfilled', # step, stepfilled
#                 linewidth=2.0,
#                 color=bar_attributes[var_ind]['color'],
#                 label=bar_attributes[var_ind]['label'],
#                 alpha=bar_attributes[var_ind]['opacity'])
#
#             x_line_arr = np.linspace(np.amin(bar_attributes[var_ind]['edges']),
#                                     np.amax(bar_attributes[var_ind]['edges']),
#                                     num=1000, endpoint=True)
#             x_line, y_line, line_label, line_color = line_function(x_line_arr, bar_arr[:, var_ind])
#             plt.plot(x_line, y_line, label=line_label, color=line_color, linewidth=2.5)
#
#         # final axis setting
#         ax.set_xlim(self.xlim)
#         ax.set_xlabel(self.xlabel, color="black")
#         ax.set_xscale('log' if self.xlog==True else 'linear')
#         ax.set_ylim(self.ylim)
#         ax.set_ylabel(self.ylabel, color="black")
#         ax.set_yscale('log' if self.ylog==True else 'linear')
#
#         # add legend
#         legend = ax.legend(loc=0)
#         legend.get_frame().set_facecolor('white')
#         legend.get_frame().set_edgecolor('lightgrey')
#
#         # save/show figure
#         if self.plot_save:
#             plt.savefig(output['output_folder'] + '/{0}.pdf'.format(output['plot_name']), bbox_inches='tight')
#         if self.plot_show:
#             plt.show(fig, block=False)
#         plt.close(fig)
#         plt.close('all')
#
#
#     def scatter(self, x_arr, y_arr, attributes, output, normalised=False):
#         """docstring for ."""
#
#         # initialise figure and axis settings
#         fig = plt.figure()
#
#         ax = plt.gca()
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#         ax.spines['bottom'].set_visible(True)
#         ax.spines['left'].set_visible(True)
#
#         # plotting of a histogram
#         plt.scatter(    x_arr, y_arr,
#                         color=attributes['color'],
#                         alpha=attributes['opacity'])
#
#         # ax.set_xticks(bins + 0.5)
#
#         # final axis setting
#         ax.set_xlim(self.xlim)
#         ax.set_xlabel(self.xlabel, color="black")
#         ax.set_xscale('log' if self.xlog==True else 'linear')
#         ax.set_ylim(self.ylim)
#         ax.set_ylabel(self.ylabel, color="black")
#         ax.set_yscale('log' if self.ylog==True else 'linear')
#
#         # add legend
#         if not attributes['label']==None:
#             legend = ax.legend(loc=0)
#             legend.get_frame().set_facecolor('white')
#             legend.get_frame().set_edgecolor('lightgrey')
#
#         # save figure
#         if self.plot_save:
#             plt.savefig(output['output_folder'] + '/{0}.pdf'.format(output['plot_name']), bbox_inches='tight')
#         if self.plot_show:
#             plt.show(fig, block=False)
#         plt.close(fig)
#         plt.close('all')
#
#     # def fig_dots_w_mult_bars(self, val_obj, attributes, legend_attr, output):
#     #     """
#     #
#     #     parameters
#     #     ----------
#     #
#     #     example
#     #     -------
#     #     x_axis = {'label': ' ',
#     #     'limits': (0, 5),
#     #     'log': False}
#     #     y_axis = {'label': 'mRNA counts',
#     #             'limits': (None, None),
#     #             'log': False}
#     #
#     #     val_obj = np.array([
#     #             np.array([[0.5, 0.45, 0.7]]),
#     #             np.array([[0.8, 0.7, 0.95], [0.4, 0.35, 0.44]]),
#     #             np.array([[0.8, 0.7, 0.95], [0.4, 0.35, 0.44], [0.4, 0.35, 0.44]]),
#     #             np.array([[0.5, 0.45, 0.7]])
#     #             ], dtype=object)
#     #
#     #     attributes = {0: ('a', ['blue']),
#     #                 1: ('b', ['limegreen', 'red']),
#     #                   2: ('b', ['blue', 'limegreen', 'red']),
#     #                 3: ('c', ['blue'])}
#     #
#     #     legend_attr = [('blue', 'x'), ('limegreen', 'y'), ('red', 'z')]
#     #
#     #     output = {'output_folder': './output',
#     #             'plot_name': 'fig_test_dots_mult_bars'}
#     #
#     #     im = pl(x_axis, y_axis, show=True)
#     #     im.fig_dots_w_mult_bars(val_obj, attributes, legend_attr, output)
#     #     """
#     #     # initialise figure and axis settings
#     #     plt.figure()
#     #
#     #     ax = plt.gca()
#     #     ax.spines['top'].set_visible(False)
#     #     ax.spines['right'].set_visible(False)
#     #     ax.spines['bottom'].set_visible(True)
#     #     ax.spines['left'].set_visible(True)
#     #
#     #     # actual plotting
#     #     for val_ind in range(val_obj.shape[0]):
#     #         color_list = attributes[val_ind][1]
#     #         dots_per_categ = val_obj[val_ind].shape[0]
#     #
#     #         if dots_per_categ == 1:
#     #             x_pos = [val_ind + 1]
#     #         elif dots_per_categ == 2:
#     #             x_pos = [val_ind + 1 - 0.2, val_ind + 1 + 0.2]
#     #         elif dots_per_categ == 3:
#     #             x_pos = [val_ind + 1 - 0.2, val_ind + 1, val_ind + 1 + 0.2]
#     #
#     #         for dot_ind in range(dots_per_categ):
#     #             plt.errorbar(x_pos[dot_ind], val_obj[val_ind][dot_ind, 0],
#     #                         yerr=val_obj[val_ind][dot_ind, 1:].reshape(2,1),
#     #                         fmt='o', capsize=4, elinewidth=2.5, markeredgewidth=2.5,
#     #                         markersize=5, markeredgecolor=color_list[dot_ind],
#     #                         color=color_list[dot_ind], ecolor='lightgrey', zorder=1000)
#     #     # comment/uncomment for grey line at zero
#     #     # plt.axhline(0, color='grey')
#     #
#     #     # final axis setting
#     #     ax.set_xlim(self.xlim)
#     #     ax.set_xlabel(self.xlabel, color="black")
#     #     ax.set_xscale('log' if self.xlog==True else 'linear')
#     #     ax.set_ylim(self.ylim)
#     #     ax.set_ylabel(self.ylabel, color="black")
#     #     ax.set_yscale('log' if self.ylog==True else 'linear')
#     #
#     #     # add x axis ticks
#     #     plt.xticks([val_ind + 1 for val_ind in range(val_obj.shape[0])],
#     #                     [attributes[val_ind][0] for val_ind in range(val_obj.shape[0])], rotation=55)
#     #
#     #     # add legend manually
#     #     plt.legend(handles=[mpatches.Patch(color=leg[0], label=leg[1]) for leg in legend_attr])
#     #
#     #     # save figure
#     #     plt.savefig(output['output_folder'] + '/{0}.pdf'.format(output['plot_name']))
#     #     if self.plot_show:
#     #         plt.show()
#     #     plt.close()
#
#
#     # def fig_hist(self, bar_arr, bins, bar_attributes, output, normalised=True):
#     #     # NOTE: this function (bin alignmight) expects float binning and looks
#     #     # weird so far in case of discrete integer binning
#     #     """
#     #     Plot a histogram for continuous values.
#     #
#     #     parameters
#     #     ----------
#     #     bar_arr
#     #         numpy array of continuous values
#     #     bins
#     #         integer, number of bins of the histogram
#     #     bar_attributes
#     #         dictionary with keys specifying a general bin 'label', bin 'color'
#     #         and how to 'align' the bins (default 'mid')
#     #     output
#     #         dictionary specifying the keys 'output_folder' and 'plot_name'
#     #
#     #     example
#     #     -------
#     #     bar_arr = np.array([1.1, 1.6, 2.1, 2.0, 0.9, 2.5, 3.3, 4.1, 3.9, 3.5, 3.6])
#     #     bins = 4
#     #     bar_attributes = {'label': 'some bins',
#     #                         'color': 'dodgerblue',
#     #                         'align': 'mid'}
#     #     output = {'output_folder': './test_figures',
#     #             'plot_name': 'fig_test_hist'}
#     #     """
#     #     # initialise figure and axis settings
#     #     plt.figure()
#     #
#     #     ax = plt.gca()
#     #     ax.spines['top'].set_visible(False)
#     #     ax.spines['right'].set_visible(False)
#     #     ax.spines['bottom'].set_visible(True)
#     #     ax.spines['left'].set_visible(True)
#     #
#     #     # plotting of a histogram
#     #     plt.hist(bar_arr, bins=bins,
#     #         normed=normalised, histtype='stepfilled',
#     #         align=bar_attributes['align'], color=bar_attributes['color'],
#     #         label=bar_attributes['label'])
#     #
#     #     # final axis setting
#     #     ax.set_xlim(self.xlim)
#     #     ax.set_xlabel(self.xlabel, color="black")
#     #     ax.set_xscale('log' if self.xlog==True else 'linear')
#     #     ax.set_ylim(self.ylim)
#     #     ax.set_ylabel(self.ylabel, color="black")
#     #     ax.set_yscale('log' if self.ylog==True else 'linear')
#     #
#     #     # add legend
#     #     legend = ax.legend(loc=0)
#     #     legend.get_frame().set_facecolor('white')
#     #     legend.get_frame().set_edgecolor('lightgrey')
#     #
#     #     # save figure
#     #     plt.savefig(output['output_folder'] + '/{0}.pdf'.format(output['plot_name']))
#     #     if self.plot_show:
#     #         plt.show()
#     #     plt.close()
