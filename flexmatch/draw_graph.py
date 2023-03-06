import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import json

def parse_data(filename: str) -> dict:
    with open(filename) as fp:
        data = fp.read().splitlines()
        parsed_data = list(map(json.loads, data))

        # aggregate w.r.t. model
        data = dict()
        for x in parsed_data:
            key = x['model'].split('/')[-1].split('.')[0]
            datum = data.get(key, [])
            datum.append(x)
            data[key] = datum
        return data
    
def draw_extraction_times(data: dict):
    # plt.clf()
    plt.title('End-to-end Term Extraction Time')
    plt.xlabel('Model')
    fig, ax = plt.subplots()
    ax.set_ylabel('Extraction Time (ms)')
    ax.set_xlabel('Model', fontsize=14)
    ax.set_title('End-to-end Term Extraction Time')
    # ax.set_yscale('log')
    x = np.arange(len(data))
    # x *= 1.5

    solver_times = dict()
    overhead_times = dict()
    x_ticks = []
    for (model, model_data) in data.items():
        x_ticks.append(model)
        for datum in model_data:
            if datum['algo'] not in solver_times:
                solver_times[datum['algo']] = []
                overhead_times[datum['algo']] = []
            solver_times[datum['algo']].append(datum['solver_time'])
            overhead_times[datum['algo']].append(datum['extract_time'] - datum['solver_time'])

    bar_width = 0.25
    ax.bar(x - bar_width, solver_times['ILP-ACyc'], bar_width, label='ILP-ACyc', color='lightgreen', edgecolor='grey')
    overhead = ax.bar(x - bar_width, overhead_times['ILP-ACyc'], bar_width, bottom=solver_times['ILP-ACyc'], color='slateblue', label='Overhead', hatch='//', edgecolor='grey')

    ax.bar(x, solver_times['WPMAXSAT'], bar_width, label='WPMAXSAT', color='lightpink', edgecolor='grey')
    overhead = ax.bar(x, overhead_times['WPMAXSAT'], bar_width, bottom=solver_times['WPMAXSAT'], color='slateblue', hatch='//', edgecolor='grey')

    ax.bar(x + bar_width, solver_times['ILP-Topo'], bar_width, label='ILP-Topo', color='lightblue', edgecolor='grey')
    overhead= ax.bar(x + bar_width, overhead_times['ILP-Topo'], bar_width, bottom=solver_times['ILP-Topo'], color='slateblue', hatch='//', edgecolor='grey')

    ax.set_xticks(x, data.keys())
    ax.set_xticklabels(x_ticks)
    ax.legend(loc='upper left')
    plt.savefig('extraction_time.png')

def draw_egraph_sizes(data: dict):
    plt.clf()
    plt.title('EGraph Stats After Equality Saturation')
    fig, (ax1, ax2) = plt.subplots(1, 2)
    num_eclasses = []
    num_enodes = []
    xticks = []
    for (model, model_data) in data.items():
        xticks.append(model)
        num_eclasses.append(model_data[0]['num_eclass'])
        num_enodes.append(model_data[0]['num_enodes'])
    tick_loc = np.arange(len(xticks))
    tick_loc *= 2
    ax1.bar(tick_loc, num_eclasses, width=1)
    ax1.set_xticks(tick_loc, xticks, rotation=90)
    ax1.set_title('Number of EClasses')
    ax2.bar(tick_loc, num_enodes, width=1)
    ax2.set_xticks(tick_loc, xticks, rotation=90)
    ax2.set_title('Number of ENodes')
    plt.tight_layout()
    plt.savefig('egraph_stats.png')

if __name__ == '__main__':
    data = parse_data('extraction_stats.txt')
    draw_extraction_times(data)
    draw_egraph_sizes(data)