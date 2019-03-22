

import numpy as np
import matplotlib.pyplot as plt
import corner
from cycler import cycler
import os


class Plots(object):
    """
    Class with multiple methods for plotting.

    methods
    -------
    fig_step_evolv
    fig_dots_w_bars_evolv
    fig_line_evolv
    fig_line_w_band_evolv
    fig_dots_w_bars_and_line_w_band_evolv
    fig_dots_w_bars
    fig_hist

    attributes
    ----------
    x_axis
        dictionary with keys specifying x axis 'label', lower and upper axis
        'limits' and 'log' scale
    y_axis
        dictionary with keys specifying y axis 'label', lower and upper axis
        'limits' and 'log' scale
    show=False (default)
        set to True to show plot

    example
    -------
    x_axis = {'label': 'time',
            'limits': (None, None),
            'log': False}
    y_axis = {'label': 'counts',
            'limits': (None, None),
            'log': False}
    """
    def __init__(self, x_axis, y_axis, show=False):
        # initialise basic information
        self.x_label = x_axis['label']
        self.x_lim = x_axis['limits']
        self.x_log = x_axis['log']

        self.y_label = y_axis['label']
        self.y_lim = y_axis['limits']
        self.y_log = y_axis['log']

        self.plot_show = show

        # update or set basic figure settings
        # NOTE: Helvetica Neue has to be installed, otherwise default font is used
        # plt.rcParams.update({'figure.autolayout': True})
        plt.rcParams.update({'figure.figsize': (8, 5)})
        plt.rcParams.update({'font.size': 14})
        plt.rcParams['font.family'] = 'Helvetica Neue'
        plt.rcParams['font.weight'] = 'medium'
        plt.rcParams['mathtext.fontset'] = 'custom'
        plt.rcParams['mathtext.rm'] = 'Helvetica Neue:medium'
        plt.rcParams['mathtext.it'] = 'Helvetica Neue:medium:italic'
        plt.rcParams['axes.labelweight'] = 'medium'
        plt.rcParams['axes.labelsize'] = 16
        plt.rcParams['axes.linewidth'] = 1.5

    def fig_network_graph(self):
        """docstring for ."""

        # TODO: implement
        pass

        # NOTE: use this somehow (arrow option for directed graphs in networkx)
        # nx.draw_networkx(G, arrows=True, **options)
        # options = {
        #     'node_color': 'blue',
        #     'node_size': 100,
        #     'width': 3,
        #     'arrowstyle': '-|>',
        #     'arrowsize': 12,
        # }


    def samples_corner(self, samples, labels, output):
        """docstring for ."""
        # initialise figure
        plt.rcdefaults()
        # plt.rcParams.update({'figure.autolayout': True})
        # plt.rcParams.update({'font.size': 16})
        plt.rcParams['font.family'] = 'Helvetica Neue'
        plt.rcParams['font.weight'] = 'medium'
        plt.rcParams['mathtext.fontset'] = 'custom'
        plt.rcParams['mathtext.rm'] = 'Helvetica Neue:medium'
        plt.rcParams['mathtext.it'] = 'Helvetica Neue:medium:italic'

        # use corner package for this plot
        corner.corner(samples, labels=labels)

        # save figure
        plt.savefig(output['output_folder'] + '/{0}.pdf'.format(output['plot_name']))
        if self.plot_show:
            plt.show()
        plt.close()


    def samples_chains(self, mcmc_sampler, num_temps, sampling_steps, num_walkers, num_params, output):
        betas = mcmc_sampler.betas
        temperatures = 1.0 / betas
        plt.rcParams['axes.titleweight'] = 'medium'
        plt.rcParams['axes.titlesize'] = 14

        for temp_ind in range(num_temps):
            plt.figure()
            plt.title(f'temp = {round(temperatures[temp_ind], 2)}, beta = {round(betas[temp_ind], 4)}') # .format(temperatures[temp_ind], betas[temp_ind]))
            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_visible(True)

            # set a color cycle for the different params
            colormap = plt.cm.viridis
            plt.gca().set_prop_cycle(cycler('color', [colormap(i) for i in np.linspace(0, 1, num_params)]))

            samples_all = mcmc_sampler.chain[temp_ind, :, :, :]
            samples_all = samples_all.reshape(sampling_steps * num_walkers, num_params)
            plt.plot(samples_all[:, :], alpha=0.75)

            # final axis setting
            ax.set_xlim(self.x_lim)
            ax.set_xlabel(self.x_label, color="black")
            ax.set_xscale('log' if self.x_log==True else 'linear')
            ax.set_ylim(self.y_lim)
            ax.set_ylabel(self.y_label, color="black")
            ax.set_yscale('log' if self.y_log==True else 'linear')

            plt.xlabel('concatenated steps of all walkers')
            plt.ylabel('parameter value')

            # save figure
            name = output['plot_name']
            plt.savefig(output['output_folder'] + f'/{name}_temp{temp_ind}.pdf')
            if self.plot_show:
                plt.show()
            plt.close()


    def fig_step_evolv(self, x_arr, y_arr, var_attributes, output):
        """
        Plot an evolving (e.g. over time) step function of multiple variables.

        parameters
        ----------
        x_arr
            numpy array with shape (# time points, )
        y_arr
            numpy array with shape (# variables, # time points)
        var_attributes
            dictionary specifying the keys 0, 1, ..., # variables - 1
            with tuples (string for legend label, plt color code for variable)
        output
            dictionary specifying the keys 'output_folder' and 'plot_name'

        example
        -------
        x_arr = np.linspace(0, 10, num=11, endpoint=True)
        y_arr = np.array([
                            [1, 1, 2, 2, 1, 2, 3, 4, 3, 3, 3],
                            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                            ])
        var_attributes = {1: ('$A_t$ (ON gene)', 'limegreen'),
                        0: ('$B_t$ (OFF gene)', 'tomato')}
        output = {'output_folder': './test_figures',
                'plot_name': 'fig_test_step'}
        """
        # initialise figure and axis settings
        plt.figure()
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)

        # actual plotting
        for var_ind in range(y_arr.shape[0]):
            var_name = var_attributes[var_ind][0]
            var_color = var_attributes[var_ind][1]

            plt.step(x_arr, y_arr[var_ind, :], label='{0}'.format(var_name),
                        where='mid', linewidth=2.75, color=var_color, alpha=0.75)
            #         plt.plot(time_arr, var_arr[var_ind, :], marker='o',
            #                     label='{0}'.format(var_name),
            #                     linewidth=2.5, markersize=6, markeredgecolor=var_color,
            #                     color=var_color, alpha=0.75)

        # final axis setting
        ax.set_xlim(self.x_lim)
        ax.set_xlabel(self.x_label, color="black")
        ax.set_xscale('log' if self.x_log==True else 'linear')
        ax.set_ylim(self.y_lim)
        ax.set_ylabel(self.y_label, color="black")
        ax.set_yscale('log' if self.y_log==True else 'linear')

        # add legend
        legend = ax.legend(loc=0)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_edgecolor('lightgrey')

        # save figure
        plt.savefig(output['output_folder'] + '/{0}.pdf'.format(output['plot_name']))
        if self.plot_show:
            plt.show()
        plt.close()


    def dots_w_bars_evolv(self, x_arr, y_arr_err, var_attributes, output):
        """
        Plot multiple evolving (e.g. over time) dots with error bars.

        parameters
        ----------
        x_arr
            numpy array with shape (# time points, )
        y_arr_err
            numpy array with shape (# variables, # time points, 2); third
            dimension includes [value, error value]
        var_attributes
            dictionary specifying the keys 0, 1, ..., # variables - 1
            with tuples (string for legend label, plt color code for variable)
        output
            dictionary specifying the keys 'output_folder' and 'plot_name'

        example
        -------
        x_arr = np.linspace(0, 2, num=3, endpoint=True)
        y_arr_err = np.array([
                            [[1, 0.1], [2, 0.2], [3, 0.3]],
                            [[2, 0.4], [1, 0.1], [4, 0.5]]
                            ])

        # use var_ind as specified in var_order
        var_attributes = {1: ('$A_t$ (ON gene)', 'limegreen'),
                        0: ('$B_t$ (OFF gene)', 'tomato')}
        output = {'output_folder': './test_figures',
                'plot_name': 'fig_test_dots_time'}
        """
        # initialise figure and axis settings
        plt.figure()

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
                        capsize=4.0, elinewidth=2.5, markeredgewidth=2.5, markersize=4.5)

        # final axis setting
        ax.set_xlim(self.x_lim)
        ax.set_xlabel(self.x_label, color="black")
        ax.set_xscale('log' if self.x_log==True else 'linear')
        ax.set_ylim(self.y_lim)
        ax.set_ylabel(self.y_label, color="black")
        ax.set_yscale('log' if self.y_log==True else 'linear')

        # add legend
        legend = ax.legend(loc=0)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_edgecolor('lightgrey')

        # save figure
        plt.savefig(output['output_folder'] + '/{0}.pdf'.format(output['plot_name']))
        if self.plot_show:
            plt.show()
        plt.close()


    def line_evolv(self, x_arr, y_line, var_attributes, output):
        """
        Plot multiple evolving (e.g. over time) lines.

        parameters
        ----------
        x_arr
            numpy array with shape (# time points, )
        y_line
            numpy array setting the multiple lines with shape (# variables, # time points)
        var_attributes
            dictionary specifying the keys 0, 1, ..., # variables - 1
            with tuples (string for legend label, plt color code for variable)
        output
            dictionary specifying the keys 'output_folder' and 'plot_name'

        example
        -------
        x_arr = np.linspace(0, 2, num=3, endpoint=True)

        y_line = np.array([
        [1, 2, 3],
        [2, 1, 4]
        ])

        var_attributes = {1: ('$A_t$ (ON gene)', 'limegreen'),
                        0: ('$B_t$ (OFF gene)', 'tomato')}
        output = {'output_folder': './test_figures',
                'plot_name': 'fig_test_line_evolv'}
        """
        # initialise figure and axis settings
        plt.figure()

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
                            color=var_color, linewidth=3, zorder=2000)

        # final axis setting
        ax.set_xlim(self.x_lim)
        ax.set_xlabel(self.x_label, color="black")
        ax.set_xscale('log' if self.x_log==True else 'linear')
        ax.set_ylim(self.y_lim)
        ax.set_ylabel(self.y_label, color="black")
        ax.set_yscale('log' if self.y_log==True else 'linear')

        # add legend
        legend = ax.legend(loc=0)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_edgecolor('lightgrey')

        # save figure
        plt.savefig(output['output_folder'] + '/{0}.pdf'.format(output['plot_name']))
        if self.plot_show:
            plt.show()
        plt.close()


    def fig_line_w_band_evolv(self, x_arr, y_line, y_lower, y_upper, var_attributes, output):
        """
        Plot multiple evolving (e.g. over time) lines with error/prediction bands.

        parameters
        ----------
        x_arr
            numpy array with shape (# time points, )
        y_line
            numpy array setting the multiple lines with shape (# variables, # time points)
        y_lower
            numpy array setting the lower bounds of the band
            with shape (# variables, # time points)
        y_upper
            numpy array setting the upper bounds of the band
            with shape (# variables, # time points)
        var_attributes
            dictionary specifying the keys 0, 1, ..., # variables - 1
            with tuples (string for legend label, plt color code for variable)
        output
            dictionary specifying the keys 'output_folder' and 'plot_name'

        example
        -------
        x_arr = np.linspace(0, 2, num=3, endpoint=True)

        y_line = np.array([
        [1, 2, 3],
        [2, 1, 4]
        ])

        y_lower = np.array([
        [0.5, 1, 2],
        [1.8, 0.8, 3.5]
        ])

        y_upper = np.array([
        [1.4, 2.7, 3.1],
        [2.1, 1.3, 4.4]
        ])

        var_attributes = {1: ('$A_t$ (ON gene)', 'limegreen'),
                        0: ('$B_t$ (OFF gene)', 'tomato')}
        output = {'output_folder': './test_figures',
                'plot_name': 'fig_test_line_band'}
        """
        # initialise figure and axis settings
        plt.figure()

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
                            color=var_color, linewidth=2, zorder=2000)

        # final axis setting
        ax.set_xlim(self.x_lim)
        ax.set_xlabel(self.x_label, color="black")
        ax.set_xscale('log' if self.x_log==True else 'linear')
        ax.set_ylim(self.y_lim)
        ax.set_ylabel(self.y_label, color="black")
        ax.set_yscale('log' if self.y_log==True else 'linear')

        # add legend
        legend = ax.legend(loc=0)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_edgecolor('lightgrey')

        # save figure
        plt.savefig(output['output_folder'] + '/{0}.pdf'.format(output['plot_name']))
        if self.plot_show:
            plt.show()
        plt.close()


    def fig_dots_w_bars_and_line_w_band_evolv(self, x_arr_dots, x_arr_line, y_dots_err, y_line, y_lower, y_upper, var_attributes, output):
        """
        Plot multiple evolving (e.g. over time) lines with error/prediction bands.

        parameters
        ----------
        x_arr_dots
            numpy array with shape (# time points, ); used for aligment with
            y_dots_err
        x_arr_line
            numpy array with shape (# time points, ); used for aligment with
            y_line, y_lower and y_upper
        y_dots_err
            numpy array with shape (# variables, # time points, 2); third
            dimension includes [value, error value]
        y_line
            numpy array setting the multiple lines with shape (# variables, # time points)
        y_lower
            numpy array setting the lower bounds of the band
            with shape (# variables, # time points)
        y_upper
            numpy array setting the upper bounds of the band
            with shape (# variables, # time points)
        var_attributes
            dictionary specifying the keys 0, 1, ..., # variables - 1
            with tuples (string for legend label, plt color code for variable)
        output
            dictionary specifying the keys 'output_folder' and 'plot_name'

        example
        -------
        x_arr_dots = np.linspace(0, 1, num=2, endpoint=True)
        x_arr_line = np.linspace(0, 2, num=3, endpoint=True)
        y_dots_err = np.array([
                            [[1, 0.1], [2, 0.3]],
                            [[2, 0.4], [1, 0.5]]
                            ])
        y_line = np.array([
        [1, 2, 3],
        [2, 1, 4]
        ])
        y_lower = np.array([
        [0.5, 1, 2],
        [1.8, 0.8, 3.5]
        ])
        y_upper = np.array([
        [1.4, 2.7, 3.1],
        [2.1, 1.3, 4.4]
        ])
        var_attributes = {1: ('$A_t$ (ON gene)', 'limegreen'),
                        0: ('$B_t$ (OFF gene)', 'tomato')}
        output = {'output_folder': './test_figures',
                'plot_name': 'fig_test_line_band_dots_bars'}
        """
        # initialise figure and axis settings
        plt.figure()

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
                            color=var_color, linewidth=2, zorder=3000)
            plt.errorbar(x_arr_dots, y_dots_err[var_ind, :, 0], yerr=y_dots_err[var_ind, :, 1],
                        fmt='o', capsize=3.5, elinewidth=2, #label='data' if var_ind==0 else '',
                        markeredgewidth=2, markersize=4, markeredgecolor='lightgrey', color='lightgrey', zorder=2000)

        # final axis setting
        ax.set_xlim(self.x_lim)
        ax.set_xlabel(self.x_label, color="black")
        ax.set_xscale('log' if self.x_log==True else 'linear')
        ax.set_ylim(self.y_lim)
        ax.set_ylabel(self.y_label, color="black")
        ax.set_yscale('log' if self.y_log==True else 'linear')

        # add legend
        legend = ax.legend(loc=0)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_edgecolor('lightgrey')

        # save figure
        plt.savefig(output['output_folder'] + '/{0}.pdf'.format(output['plot_name']))
        if self.plot_show:
            plt.show()
        plt.close()


    # # TODO: steady state plots / volume plots?
    # def fig_dots_w_bars(self, y_arr_err, attributes, output, x_ticks=None):
    #     """
    #     Plot dots with error bars (values in y axis, iteration over x axis).
    #
    #     parameters
    #     ----------
    #     y_arr_err
    #         numpy array with shape (# dots, 2); errors are given in the second
    #         dimension
    #     attributes
    #         dictionary specifying the keys 'color'
    #     output
    #         dictionary specifying the keys 'output_folder' and 'plot_name'
    #     x_ticks (optional)
    #         list of labels for dots which are plotted below x axis.
    #
    #     example
    #     -------
    #     y_arr_err = np.array([
    #     [1, 0.2],
    #     [2, 0.8],
    #     [3, 0.3]
    #     ])
    #
    #     x_ticks = ['a', 'b', 'c']
    #
    #     attributes = {'color': 'dodgerblue'}
    #
    #     output = {'output_folder': './test_figures',
    #             'plot_name': 'fig_test_dots_bars'}
    #     """
    #     # initialise figure and axis settings
    #     plt.figure()
    #
    #     ax = plt.gca()
    #     ax.spines['top'].set_visible(False)
    #     ax.spines['right'].set_visible(False)
    #     ax.spines['bottom'].set_visible(True)
    #     ax.spines['left'].set_visible(True)
    #
    #     # actual plotting
    #     dot_color = attributes['color']
    #     for dot_ind in range(y_arr_err.shape[0]):
    #         plt.errorbar(dot_ind + 1, y_arr_err[dot_ind, 0], yerr=y_arr_err[dot_ind, 1],
    #                     fmt='o', capsize=4, elinewidth=2.5, markeredgewidth=2.5,
    #                     markersize=5, markeredgecolor=dot_color, color=dot_color, ecolor='lightgrey')
    #
    #     # final axis setting
    #     ax.set_xlim(self.x_lim)
    #     ax.set_xlabel(self.x_label, color="black")
    #     ax.set_xscale('log' if self.x_log==True else 'linear')
    #     ax.set_ylim(self.y_lim)
    #     ax.set_ylabel(self.y_label, color="black")
    #     ax.set_yscale('log' if self.y_log==True else 'linear')
    #
    #     if x_ticks != None:
    #         plt.xticks([i + 1 for i in range(y_arr_err.shape[0])],
    #                     [x_ticks[i] for i in range(y_arr_err.shape[0])], rotation=55)
    #
    #     # save figure
    #     plt.savefig(output['output_folder'] + '/{0}.pdf'.format(output['plot_name']))
    #     if self.plot_show:
    #         plt.show()
    #     plt.close()


    def dots_w_bars(self, y_arr_err, x_ticks, attributes, output):
        """
        Plot dots with error bars (values in y axis, iteration over x axis).

        parameters
        ----------
        y_arr_err
            numpy array with shape (# dots, 2); errors are given in the second
            dimension
        attributes
            dictionary specifying the keys 'color'
        x_ticks (set None to ignore)
            list of labels for dots which are plotted below x axis.
        output
            dictionary specifying the keys 'output_folder' and 'plot_name'


        example
        -------
        y_arr_err = np.array([
        [1, 0.2],
        [2, 0.8],
        [3, 0.3]
        ])

        x_ticks = ['a', 'b', 'c']

        attributes = {'color': 'dodgerblue'}

        output = {'output_folder': './test_figures',
                'plot_name': 'fig_test_dots_bars'}
        """

        # initialise figure and axis settings
        plt.figure()

        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)

        # actual plotting
        for dot_ind in range(y_arr_err.shape[0]):
            plt.errorbar(dot_ind + 1, y_arr_err[dot_ind, 0],
                        yerr=np.array([y_arr_err[dot_ind, 1], y_arr_err[dot_ind, 2]]).reshape(2,1),
                        fmt='o', capsize=4, elinewidth=2.5, markeredgewidth=2.5,
                        markersize=5, markeredgecolor=attributes[dot_ind][1],
                        color=attributes[dot_ind][1], ecolor='lightgrey')

        # final axis setting
        ax.set_xlim(self.x_lim)
        ax.set_xlabel(self.x_label, color="black")
        ax.set_xscale('log' if self.x_log==True else 'linear')
        ax.set_ylim(self.y_lim)
        ax.set_ylabel(self.y_label, color="black")
        ax.set_yscale('log' if self.y_log==True else 'linear')

        if x_ticks != None:
            plt.xticks([i + 1 for i in range(y_arr_err.shape[0])],
                        [x_ticks[i] for i in range(y_arr_err.shape[0])], rotation=55)

        # save figure
        plt.savefig(output['output_folder'] + '/{0}.pdf'.format(output['plot_name']))
        if self.plot_show:
            plt.show()
        plt.close()


    def fig_dots_w_mult_bars(self, val_obj, attributes, legend_attr, output):
        """

        parameters
        ----------

        example
        -------
        x_axis = {'label': ' ',
        'limits': (0, 5),
        'log': False}
        y_axis = {'label': 'mRNA counts',
                'limits': (None, None),
                'log': False}

        val_obj = np.array([
                np.array([[0.5, 0.45, 0.7]]),
                np.array([[0.8, 0.7, 0.95], [0.4, 0.35, 0.44]]),
                np.array([[0.8, 0.7, 0.95], [0.4, 0.35, 0.44], [0.4, 0.35, 0.44]]),
                np.array([[0.5, 0.45, 0.7]])
                ], dtype=object)

        attributes = {0: ('a', ['blue']),
                    1: ('b', ['limegreen', 'red']),
                      2: ('b', ['blue', 'limegreen', 'red']),
                    3: ('c', ['blue'])}

        legend_attr = [('blue', 'x'), ('limegreen', 'y'), ('red', 'z')]

        output = {'output_folder': './output',
                'plot_name': 'fig_test_dots_mult_bars'}

        im = pl(x_axis, y_axis, show=True)
        im.fig_dots_w_mult_bars(val_obj, attributes, legend_attr, output)
        """
        # initialise figure and axis settings
        plt.figure()

        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)

        # actual plotting
        for val_ind in range(val_obj.shape[0]):
            color_list = attributes[val_ind][1]
            dots_per_categ = val_obj[val_ind].shape[0]

            if dots_per_categ == 1:
                x_pos = [val_ind + 1]
            elif dots_per_categ == 2:
                x_pos = [val_ind + 1 - 0.2, val_ind + 1 + 0.2]
            elif dots_per_categ == 3:
                x_pos = [val_ind + 1 - 0.2, val_ind + 1, val_ind + 1 + 0.2]

            for dot_ind in range(dots_per_categ):
                plt.errorbar(x_pos[dot_ind], val_obj[val_ind][dot_ind, 0],
                            yerr=val_obj[val_ind][dot_ind, 1:].reshape(2,1),
                            fmt='o', capsize=4, elinewidth=2.5, markeredgewidth=2.5,
                            markersize=5, markeredgecolor=color_list[dot_ind],
                            color=color_list[dot_ind], ecolor='lightgrey', zorder=1000)
        # comment/uncomment for grey line at zero
        # plt.axhline(0, color='grey')

        # final axis setting
        ax.set_xlim(self.x_lim)
        ax.set_xlabel(self.x_label, color="black")
        ax.set_xscale('log' if self.x_log==True else 'linear')
        ax.set_ylim(self.y_lim)
        ax.set_ylabel(self.y_label, color="black")
        ax.set_yscale('log' if self.y_log==True else 'linear')

        # add x axis ticks
        plt.xticks([val_ind + 1 for val_ind in range(val_obj.shape[0])],
                        [attributes[val_ind][0] for val_ind in range(val_obj.shape[0])], rotation=55)

        # add legend manually
        plt.legend(handles=[mpatches.Patch(color=leg[0], label=leg[1]) for leg in legend_attr])

        # save figure
        plt.savefig(output['output_folder'] + '/{0}.pdf'.format(output['plot_name']))
        if self.plot_show:
            plt.show()
        plt.close()


    def fig_hist(self, bar_arr, bins, bar_attributes, output, normalised=True):
        # NOTE: this function (bin alignmight) expects float binning and looks
        # weird so far in case of discrete integer binning
        """
        Plot a histogram for continuous values.

        parameters
        ----------
        bar_arr
            numpy array of continuous values
        bins
            integer, number of bins of the histogram
        bar_attributes
            dictionary with keys specifying a general bin 'label', bin 'color'
            and how to 'align' the bins (default 'mid')
        output
            dictionary specifying the keys 'output_folder' and 'plot_name'

        example
        -------
        bar_arr = np.array([1.1, 1.6, 2.1, 2.0, 0.9, 2.5, 3.3, 4.1, 3.9, 3.5, 3.6])
        bins = 4
        bar_attributes = {'label': 'some bins',
                            'color': 'dodgerblue',
                            'align': 'mid'}
        output = {'output_folder': './test_figures',
                'plot_name': 'fig_test_hist'}
        """
        # initialise figure and axis settings
        plt.figure()

        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)

        # plotting of a histogram
        plt.hist(bar_arr, bins=bins,
            normed=normalised, histtype='stepfilled',
            align=bar_attributes['align'], color=bar_attributes['color'],
            label=bar_attributes['label'])

        # final axis setting
        ax.set_xlim(self.x_lim)
        ax.set_xlabel(self.x_label, color="black")
        ax.set_xscale('log' if self.x_log==True else 'linear')
        ax.set_ylim(self.y_lim)
        ax.set_ylabel(self.y_label, color="black")
        ax.set_yscale('log' if self.y_log==True else 'linear')

        # add legend
        legend = ax.legend(loc=0)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_edgecolor('lightgrey')

        # save figure
        plt.savefig(output['output_folder'] + '/{0}.pdf'.format(output['plot_name']))
        if self.plot_show:
            plt.show()
        plt.close()
