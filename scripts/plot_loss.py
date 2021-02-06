#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import argparse


def load_data( data_file ):
    with open(data_file, 'r') as f:
        lines = f.readlines()
        legend = lines[0]
        legend = legend[:-2].split(',')
        num_var = len(legend)
        print 'Number of variables: ', num_var

        legend_dictionary = {}
        for index, item in enumerate(legend):
            legend_dictionary[item] = index

        variables = []
        num_lines = len(lines)
        for i in range(1, num_lines):
            variables.append( lines[i][:-2].split(',') )

    variables = np.asarray( variables, dtype=np.float )
    return legend_dictionary, variables


def plot_data( legend_dictionary, variables ):
    x_axis = legend_dictionary['NumIters']
    y_axis = legend_dictionary['loss']

    x_data = variables[:,x_axis]
    y_data = variables[:,y_axis]


    plt.subplot(211)
    plt.semilogy(x_data, y_data, 'k', x_data, y_data, 'bo')
    plt.title('semilogy')

    plt.subplot(212)
    plt.plot( x_data, y_data, 'bo', x_data, y_data, 'k')
    plt.title('linear scale')
    plt.show()
    return



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--log', type=str, help='path to log file')
    args = parser.parse_args()
    log_file = args.log

    legend_dictionary, variables = load_data( log_file )
    plot_data( legend_dictionary, variables )


