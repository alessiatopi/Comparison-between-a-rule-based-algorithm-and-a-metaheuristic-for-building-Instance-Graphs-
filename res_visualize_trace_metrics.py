from os.path import join
from os import makedirs
import pandas as pd
from matplotlib import pyplot as plt
from config import COMBINATIONS, MAX_LIMIT_NUM_TRACES, PLOTS_PATH, ITERATION_RESULTS_PATH, SEEDS, get_comb_parameters


def visualize_trace_metric(trace_id, iterations, values, seed, metric, params, path):
    makedirs(path, exist_ok=True)

    plt.plot(iterations, values)
    plt.xlabel('Iterations')
    plt.ylabel(metric)
    plt.xticks([i for i in range(-1, len(iterations)+1, 5)])
    plt.title(f'Trace {trace_id}, seed {seed}')

    if metric == 'Generalization':
        plt.yticks([i for i in range(0, MAX_LIMIT_NUM_TRACES+1, MAX_LIMIT_NUM_TRACES)])
    """
    if metric == 'fo':
        plt.yticks(numpy.arange(0, 5, 0.5))
    """

    plt.savefig(join(path, f"{trace_id}_{params}_s{seed}.png"))
    plt.close('all')


def process():
    for comb in COMBINATIONS:
        iters = comb['iterations']
        params = get_comb_parameters(comb)
        print(f'\nProcessing comb: {params}')

        df = pd.read_csv(join(ITERATION_RESULTS_PATH, f"iteration_results_{params}.csv"), header=0, sep=';')
        seeds = sorted(df['seed'].unique().tolist())
        trace_ids = sorted(df['trace_id'].unique().tolist())
        iterations = [i for i in range(-1, iters)]

        for seed in sorted(SEEDS):
            print(f'Processing seed: {seed}')
            filtered_seed = df.loc[(df['seed'] == seed)]

            for trace_id in trace_ids:
                print(f'\tProcessing trace: {trace_id}')
                filtered_seed_trace = filtered_seed.loc[(filtered_seed['trace_id'] == trace_id)]

                for metric in ['gen', 'fo_curr']:
                    m = 'Generalization' if metric == 'gen' else "Objective function"
                    path = join(PLOTS_PATH, m, f'{seed}', params)
                    values = filtered_seed_trace[metric].tolist()
                    visualize_trace_metric(trace_id, iterations, values, seed, m, params, path)


if __name__ == '__main__':
    process()

