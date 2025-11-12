import pandas as pd
from os.path import join
from config import get_comb_parameters, COMBINATIONS, OUTPUT_PATH, ITERATION_RESULTS_PATH, SEEDS


if __name__ == '__main__':
    recap_df = pd.DataFrame(columns=["combination", "seed", "iteration", "trace_id", "total_times_sim_all",
                                     "positive_contribution", "best_before_sim_all", "best_until_next_sim_all"])
    for comb in COMBINATIONS:
        iters = comb['iterations']
        comb_parameters = get_comb_parameters(comb)
        iteration_data = pd.read_csv(join(ITERATION_RESULTS_PATH, f"iteration_results_{comb_parameters}.csv"), sep=';', header=0)

        for seed in sorted(SEEDS):
            seed_data = iteration_data[iteration_data['seed'] == seed]
            trace_ids = sorted(seed_data['trace_id'].unique().tolist())

            for trace in trace_ids:
                trace_seed_data = seed_data[seed_data['trace_id'] == trace]
                # filtriamo le righe dalle iterazioni in cui l'ig non fitta la traccia
                # perchè non vengono effettuati i calcoli delle funzioni obiettivo
                trace_seed_data = trace_seed_data[trace_seed_data['log_meta'] != 'no_fit']

                times_sim_all = trace_seed_data['result']
                # prendiamo le righe dove viene fatto simulated annealing
                times_sim_all = times_sim_all[times_sim_all == 'new one accepted (simulated annealing)']

                # contiamo il numero di volte che viene fatto simulated annealing
                total_times_sim_all = len(times_sim_all)

                times_sim_all_indexes = times_sim_all.index
                for ix_start, ix_end in zip(times_sim_all_indexes, times_sim_all_indexes[1:]):

                    # valore della fo e iterazione in cui viene fatto simulated annealing
                    fo_before_sim_all = trace_seed_data.loc[ix_start, 'fo_curr']
                    iteration_sim_all = trace_seed_data.loc[ix_start, 'iteration']

                    # prendiamo le iterazioni che occorrono tra due simulated annealing
                    fragment = trace_seed_data.loc[ix_start + 1:ix_end]

                    # prendiamo il valore minimo della fo nel frammento
                    best_fo_after_sim_all = fragment.loc[fragment['fo_curr'] ==
                                                         fragment['fo_curr'].min(), 'fo_curr'].tolist()[0]

                    # si ha un contributo positivo del simulated annealing quando il valore della fo che troviamo
                    # nel frammento [ix_start + 1:ix_end] è minore rispetto a quello nell'iterazione
                    # in cui avviene il simulated annealing [ix_start]
                    positive_contribution = True if best_fo_after_sim_all < fo_before_sim_all else False
                    recap_df.loc[len(recap_df)] = [comb_parameters, seed, iteration_sim_all, trace, total_times_sim_all,
                                                   positive_contribution, fo_before_sim_all, best_fo_after_sim_all]

    # aggreghiamo i valori precedenti per ogni combinazione sommando il numero totale di contributi positivi e negativi
    agg = recap_df.groupby(['combination', 'seed'])['positive_contribution'].agg(
        positive_contribution=lambda x: x.sum(),
        negative_contribution=lambda x: (~x).sum())
    agg.to_csv(join(OUTPUT_PATH, 'sim_ann_performance.csv'), index=True, sep=';')
