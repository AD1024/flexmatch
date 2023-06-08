import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
import numpy as np
import json

def parse_data(filename: str) -> dict:
    with open(filename) as fp:
        data = fp.read().splitlines()
        parsed_data = list(map(json.loads, data))

        # aggregate w.r.t. model
        data = dict()
        for x in parsed_data:
            key = x['model'].split('/')[-1].split('.')[0].split('_')[0]
            datum = data.get(key, [])
            datum.append(x)
            data[key] = datum
        return data
    
def draw_extraction_times(data: dict, simpl: bool):
    # plt.clf()
    plt.xlabel('Model')
    fig, ax = plt.subplots()
    ax.set_ylabel('Extraction Time (ms)', fontsize=14)
    # ax.set_xlabel('Model', fontsize=14)
    ax.set_title(r'\begin{center}(\textsc{im2col} + \textsc{simpl})\end{center}' if simpl else r'\begin{center}Term Extraction Time\\(\textsc{im2col} only)\end{center}', fontsize=14)
    # ax.set_yscale('log')
    x = np.arange(len(data))
    # x *= 1.5

    solver_times = dict()
    overhead_times = dict()
    x_ticks = []
    for (model, model_data) in data.items():
        i = len(x_ticks)
        x_ticks.append(model)
        counter = dict()
        for datum in model_data:
            solver_time = datum['solver_time']
            overhead_time = datum['extract_time'] - datum['solver_time']
            counter[datum['algo']] = counter.get(datum['algo'], 0)
            if datum['algo'] not in solver_times:
                solver_times[datum['algo']] = {}
                overhead_times[datum['algo']] = {}
            if counter[datum['algo']] == 0:
                solver_times[datum['algo']][i] = solver_time
                overhead_times[datum['algo']][i] = overhead_time
            else:
                solver_times[datum['algo']][i] += solver_time
                overhead_times[datum['algo']][i] += overhead_time
            counter[datum['algo']] += 1
        for (algo, count) in counter.items():
            solver_times[algo][i] /= count
            overhead_times[algo][i] /= count

    bar_width = 0.25
    ilp_acyc_xaxis = np.array(sorted(list(solver_times['ILP-ACyc'].keys())))
    ilp_acyc_solver_times = list(map(solver_times['ILP-ACyc'].get, ilp_acyc_xaxis))
    ilp_acyc_overhead_times = list(map(overhead_times['ILP-ACyc'].get, ilp_acyc_xaxis))
    ax.bar(ilp_acyc_xaxis, ilp_acyc_solver_times, bar_width, label='ILP-ACyc', color='lightgreen', edgecolor='grey')
    overhead = ax.bar(ilp_acyc_xaxis, ilp_acyc_overhead_times, bar_width, bottom=ilp_acyc_solver_times, color='slateblue', label='Overhead', hatch='//', edgecolor='grey')

    maxsat_xaxis = np.array(sorted(list(solver_times['WPMAXSAT'].keys())))
    maxsat_solver_times = list(map(solver_times['WPMAXSAT'].get, maxsat_xaxis))
    maxsat_overhead_times = list(map(overhead_times['WPMAXSAT'].get, maxsat_xaxis))
    print('Maxsat times:', maxsat_solver_times, maxsat_overhead_times)
    ax.bar(maxsat_xaxis - bar_width, maxsat_solver_times, bar_width, label='WPMAXSAT', color='lightblue', edgecolor='grey')
    overhead = ax.bar(maxsat_xaxis - bar_width, maxsat_overhead_times, bar_width, bottom=maxsat_solver_times, color='slateblue', hatch='//', edgecolor='grey')

    ilp_topo_xaxis = np.array(sorted(list(solver_times['ILP-Topo'].keys())))
    ilp_topo_solver_times = list(map(solver_times['ILP-Topo'].get, ilp_topo_xaxis))
    ilp_topo_overhead_times = list(map(overhead_times['ILP-Topo'].get, ilp_topo_xaxis))
    ax.bar(ilp_topo_xaxis + bar_width, ilp_topo_solver_times, bar_width, label='ILP-Topo', color='lightpink', edgecolor='grey')
    overhead= ax.bar(ilp_topo_xaxis + bar_width, ilp_topo_overhead_times, bar_width, bottom=ilp_topo_solver_times, color='slateblue', hatch='//', edgecolor='grey')

    labeled = False
    for i in ilp_acyc_xaxis:
        if i not in ilp_topo_xaxis:
            ax.bar([i], [0])
            if not labeled:
                ax.axvline(i + 1.25 * bar_width, color='red', linestyle='--', label='ILP-Topo timeouts (20s)')
                labeled = True
            else:
                ax.axvline(i + 1.25 * bar_width, color='red', linestyle='--')

    try:
        for i in ilp_topo_xaxis:
            print((np.array(solver_times['ILP-Topo'][i]) + overhead_times['ILP-Topo'][i]) / (np.array(solver_times['ILP-ACyc'][i]) + overhead_times['ILP-ACyc'][i]))
            print((np.array(solver_times['ILP-Topo'][i]) + overhead_times['ILP-Topo'][i]) / (np.array(solver_times['WPMAXSAT'][i]) + overhead_times['WPMAXSAT'][i]))
    except:
        print('oopse')
    # print(solver_times['ILP-ACyc'])
    # print(solver_times['WPMAXSAT'])
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='both', which='minor', labelsize=15)

    if simpl:
        ax.set_xticks(x, data.keys())
        ax.set_xticklabels(x_ticks)
        ax.set_yscale('symlog', base=2)
    else:
        ax.set_xticks([], [])

    if simpl:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                    box.width, box.height * 0.9])

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
            fancybox=False, shadow=True, ncol=2, fontsize=14)
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
    #       ncol=3, fancybox=True, shadow=True)
    plt.savefig('extraction_time_im2col.pdf' if not simpl else 'extraction_time_im2col_simpl.pdf', bbox_inches='tight')

def draw_egraph_sizes(data: dict, simpl: bool):
    plt.clf()
    plt.title(r'EGraph Stats After Equality Saturation (\textsc{im2col} only)' if not simpl else r'EGraph Stats After Equality Saturation (\textsc{im2col} + \textsc{simpl})')
    fig, (ax1, ax2) = plt.subplots(1, 2)
    num_eclasses = []
    num_enodes = []
    xticks = []
    for (model, model_data) in data.items():
        xticks.append(model)
        num_eclasses.append(model_data[0]['num_eclass'])
        num_enodes.append(model_data[0]['num_enodes'])
    print(num_eclasses)
    print(num_enodes)
    tick_loc = np.arange(len(xticks))
    tick_loc *= 2
    ax1.bar(tick_loc, num_eclasses, width=1, color='mediumslateblue')
    if simpl:
        ax1.set_xticks(tick_loc, xticks, rotation=90)
    else:
        ax1.set_xticks([], [])
    ax1.set_title(r'\begin{center}Number of EClasses\\(\textsc{im2col} only)\end{center}' if not simpl else r'\begin{center}(\textsc{im2col} + \textsc{simpl})\end{center}', fontsize=16)
    ax1.get_title()
    ax2.bar(tick_loc, num_enodes, width=1, color='mediumslateblue')
    if simpl:
        ax2.set_xticks(tick_loc, xticks, rotation=90)
    else:
        ax2.set_xticks([], [])
    ax2.set_title(r'\begin{center}Number of ENodes\\(\textsc{im2col} only)\end{center}' if not simpl else r'\begin{center}(\textsc{im2col} + \textsc{simpl})\end{center}', fontsize=16)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax1.tick_params(axis='both', which='minor', labelsize=14)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    ax2.tick_params(axis='both', which='minor', labelsize=14)
    plt.tight_layout()
    plt.savefig('egraph_stats_im2col.pdf' if not simpl else 'egraph_stats_im2col_simpl.pdf')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('extraction_stats', type=str, help='extraction stats file')
    parser.add_argument('--render-simpl', action='store_true')
    args = parser.parse_args()
    data = parse_data(args.extraction_stats)
    draw_extraction_times(data, args.render_simpl)
    draw_egraph_sizes(data, args.render_simpl)