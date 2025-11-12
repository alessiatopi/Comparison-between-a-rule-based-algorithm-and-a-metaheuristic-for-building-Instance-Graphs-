import pandas as pd
from os.path import join
from config import (get_comb_parameters, TRACES_MIN_FO_PATH, COMBINATIONS, BIG_TIMES_FILE_PATH, OUTPUT_PATH,
                    TRACE_RESULTS_PATH, ITERATION_RESULTS_PATH, SEEDS, MAX_LIMIT_NUM_TRACES)




def process():
    big_times = pd.read_csv(BIG_TIMES_FILE_PATH, sep=';', header=0)
    recap_df = pd.DataFrame(columns=["comb", "seed", "gen_smaller_big", "gen_smaller_meta", "gen_equal",
                                     "MSE_gen_filtered", "struct_move_smaller_big", "struct_move_smaller_meta",
                                     "struct_move_equal",  "align_move_smaller_big", "align_move_smaller_meta", "align_move_equal",
                                     "meta_failures", "avg_num_edges_big", "avg_num_edges_meta", "time_big",
                                     "time_meta", "avg_min_iteration_fo"])
    for comb in COMBINATIONS:
        comb_parameters = get_comb_parameters(comb)
        meta_iterations = pd.read_csv(join(ITERATION_RESULTS_PATH, f"iteration_results_{comb_parameters}.csv"), sep=';', header=0)

        traces_min_fo_iteration_df = pd.DataFrame([], columns=["seed", "trace_id", "min_iteration"])
        for trace_id in meta_iterations["trace_id"].unique().tolist():
            trace_df = meta_iterations.loc[(meta_iterations['trace_id'] == trace_id)]
            for seed in SEEDS:
                seed_df = trace_df.loc[(trace_df['seed'] == seed)]
                try:
                    min_fo_nows = seed_df.loc[(seed_df['fo_now'] == seed_df['fo_now'].min()), 'iteration'].tolist()[0]
                    min_fo_currs = seed_df.loc[(seed_df['fo_curr'] == seed_df['fo_curr'].min()), 'iteration'].tolist()[0]
                    if min_fo_currs < min_fo_nows:
                        traces_min_fo_iteration_df.loc[len(traces_min_fo_iteration_df)] = [seed, trace_id, min_fo_currs]
                    else:
                        traces_min_fo_iteration_df.loc[len(traces_min_fo_iteration_df)] = [seed, trace_id, min_fo_nows]
                except IndexError:
                    print(f"No fo found for combination: {comb}, seed: {seed}, trace: {trace_id}")

        traces_min_fo_iteration_df.to_csv(join(TRACES_MIN_FO_PATH, f"traces_min_fo_iteration_{comb_parameters}.csv"))
        avg_iteration_min_seeds = traces_min_fo_iteration_df[['seed', 'min_iteration']].groupby(['seed']).mean().reset_index()

        trace_results = pd.read_csv(join(TRACE_RESULTS_PATH, f"trace_results_{comb_parameters}.csv"), sep=';', header=0)
        for seed in sorted(SEEDS):
            seed_traces = trace_results[trace_results['seed'] == seed]

            tot_time_big, tot_time_meta = float(big_times["time"].sum()), float(seed_traces["time_meta"].sum())

            smaller_big_gen = len(seed_traces.loc[seed_traces['gen_big_rep'] < seed_traces['gen_meta_best']])
            smaller_meta_gen = len(seed_traces.loc[seed_traces['gen_big_rep'] > seed_traces['gen_meta_best']])
            equal_gen = len(seed_traces.loc[seed_traces['gen_big_rep'] == seed_traces['gen_meta_best']])

            # nel calcolo dell'mse escludiamo i limiti impostati nel file di configurazione, altrimenti la metrica sballa
            filtered_gen_mse = seed_traces.loc[(seed_traces['gen_meta_best'] != MAX_LIMIT_NUM_TRACES)]
            mse = float(((filtered_gen_mse['gen_big_rep'] - filtered_gen_mse['gen_meta_best']) ** 2).mean())

            meta_success = seed_traces.loc[seed_traces['best_meta_struct_move'].notna()]
            smaller_big_move = len(meta_success.loc[meta_success['big_rep_struct_move'] - meta_success['best_meta_struct_move'] < 0])
            smaller_meta_move = len(meta_success.loc[meta_success['big_rep_struct_move'] - meta_success['best_meta_struct_move'] > 0])
            equal_move = len(meta_success.loc[meta_success['big_rep_struct_move'] - meta_success['best_meta_struct_move'] == 0])
            meta_fail_move = len(seed_traces) - len(meta_success)

            meta_success = seed_traces.loc[seed_traces['best_meta_align_move'].notna()]

            smaller_big_align_move = len(meta_success.loc[meta_success['big_rep_align_move'] - meta_success['best_meta_align_move'] < 0])
            smaller_meta_align_move = len(meta_success.loc[meta_success['big_rep_align_move'] - meta_success['best_meta_align_move'] > 0])
            equal_align_move = len(meta_success.loc[meta_success['big_rep_align_move'] - meta_success['best_meta_align_move'] == 0])


            avg_edges_big = float(seed_traces['num_edges_big_rep'].mean())
            avg_edges_meta = float(meta_success['num_edges_best_meta'].mean())

            recap_df.loc[len(recap_df)] = [comb_parameters, seed, smaller_big_gen, smaller_meta_gen, equal_gen,
                                           mse, smaller_big_move, smaller_meta_move,
                                           equal_move, smaller_big_align_move, smaller_meta_align_move,
                                           equal_align_move, meta_fail_move, avg_edges_big, avg_edges_meta,
                                           tot_time_big, tot_time_meta,
                                           avg_iteration_min_seeds.loc[avg_iteration_min_seeds['seed'] == seed, 'min_iteration'].item()]

    recap_df.to_csv(join(OUTPUT_PATH, 'compare_combinations.csv'), index=False, sep=';')


if __name__ == '__main__':
    process()