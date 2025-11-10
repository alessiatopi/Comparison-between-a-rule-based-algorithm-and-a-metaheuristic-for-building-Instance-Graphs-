from os.path import join
from pm4py.objects.log.importer.xes.variants import iterparse as xes_importer
from pandas import DataFrame
from big import find_causal_relationships, apply_big, find_dependency_net, apply_conformance
from meta import *
from utils import get_process_model, save_and_draw_ig, read_ig
from res_combination_seeds import process as evaluate_results
from res_visualize_trace_metrics import process as visualize_fos
from config import (COMBINATIONS, LOG_FILE_PATH, IGS_PATH,
                    SEEDS, MAX_LIMIT_NUM_TRACES, get_comb_parameters, BIG_IGS_PATH,
                    TIMES_PATH, TRACE_RESULTS_PATH, ITERATION_RESULTS_PATH)
'''
CHANGELOG:
- quando terminano tutte le combinazioni, esegue direttamente res_combination_seeds e res_visualize_metrics
(la seconda al momento è commentata per motivi di tempo), ma almeno non devi ogni volta rilanciarlo a parte
- aggiunta la fase di accettazione della soluzione iniziale con le relative statistiche nei file 'iterations', l'iterazione
corrispondente ha come indice (-1)
- modifica di alcuni nomi dei campi per renderli più chiari
- modificato comportamento della procedura di simulazione (simulator_apply)
- la generalizzazione massima viene definita a runtime facendo una simulazione sul ptree derivato. Se va in 
timeout si setta il threshold trovato in config; se riesce correttamente, la modifica deve essere necessariamente
propagata quando si calcolano le variabili della funzione obiettivo. Prima la seconda casistica non succedeva, 
quindi in funzione obiettivo la componente dipendente dalla generalizzazione assumeva valori con una scala
molto maggiore rispetto alla componente relativa alle mosse
ps. non ti preoccupare delle mosse negative, non è un problema
- per verificare quando la metaeuristica si comporta meglio di big riparato, ora hai altri dati salvati per ogni traccia (trace_results)
quali i valori della funzione obiettivo di best meta e di big riparato, incluso l'elenco degli archi di entrambi.
Quando determini la soluzione vincente devi controllare oltre al valore di fo (che potenzialmente potrebbe essere uguale per entrambi)
anche la lista degli archi di entrambe perchè potrebbero essere diverse (big e meta magari si comportano numericamente uguale ma 
gli archi potrebbero essere diversi, e sarebbe interessante in caso
"""

'''
def main():
    # READ LOG AND PETRI NET
    print('Reading log and petri net...')
    initial_time = time()
    log = xes_importer.import_log(LOG_FILE_PATH)
    process_model, model_activities = get_process_model(log)
    net, init_marking, final_marking = process_model[0], process_model[1], process_model[2]

    tree = wf_net_converter.apply(net, init_marking, final_marking)
    param = tree_playout.Variants.EXTENSIVE.value.Parameters
    simulated_log = tree_playout.apply(tree, variant=tree_playout.Variants.EXTENSIVE,
                                       parameters={param.MAX_TRACE_LENGTH: MAX_TRACE_LENGTH,
                                                   param.MAX_LIMIT_NUM_TRACES: MAX_LIMIT_NUM_TRACES})
    gen_max = len(simulated_log)
    # -----

    # EXTRACT CAUSAL RELATIONS
    print('Discovering causal relations...')
    causal_rels = find_causal_relationships(net)
    causal_rels = sorted(causal_rels, key=lambda x: x[0])
    model_dep = find_dependency_net(causal_rels, net)
    # -----

    # CONFORMANCE CHECKING AND BIG
    aligned_igs = apply_conformance(log, process_model, causal_rels, model_activities)
    apply_big(log, process_model, aligned_igs, causal_rels) #commento per il debug veloce
    # -----

    # STARTING COMBINATIONS
    for id_comb, comb in enumerate(COMBINATIONS):

        # CREATE DFs FOR CURRENT COMBINATION
        time_results = DataFrame(columns=["comb", "seed", "trace", "time"])
        trace_results = DataFrame(columns=["seed", "trace_id", "num_nodes", "time_meta",
                                           "gen_meta_init", "log_meta_init", "gen_meta_best", "log_meta_best",
                                           "best_meta_struct_move", "best_meta_align_move", "fo1_best_meta",
                                           "fo2_best_meta", "fo_meta_best", "counter_no_process_tree",
                                           "counter_sim_all", "best_solutions", "num_edges_best_meta",
                                           "edges_best_meta",  "num_edges_big_rep", "edges_big_rep", "gen_big_rep",
                                           "big_rep_struct_move", "big_rep_align_move", "fo1_big_rep", "fo2_big_rep", "fo_big_rep"])
        it_results = DataFrame(columns=["seed", "trace_id", "meta_init", "iteration", "log_meta", "fo_curr",
                                        "fo_now", "fo1", "fo2", "result", "gen", "struct_move", "align_move",
                                        "time"] + [f'move_local_search_{i}' for i in range(comb['k'])])
        # -----

        string_comb_parameters = get_comb_parameters(comb)
        print(f'Starting combination: {string_comb_parameters} ({id_comb + 1}/{len(COMBINATIONS)})')
        remaining_seeds = len(SEEDS)
        for seed in sorted(SEEDS):
            random.seed(seed)
            comb_start_time = time()
            print(f'*** Seed: {seed} ({remaining_seeds} remaining) ***')
            remaining_seeds -= 1

            remaining_traces = len(aligned_igs)
            for case_id, edges_aligned in zip(aligned_igs['case_id'].tolist(), aligned_igs['edges_aligned'].tolist()):
                remaining_traces -= 1
                print(f"\n\n*** Processing case: {case_id} ({remaining_traces} remaining) ***")

                _, edges_initial_big = read_ig(join(BIG_IGS_PATH, f'{case_id}_big_no_repaired'))
                #print(f"[DEBUG] Traccia {case_id} - IG iniziale spurio (big_no_repaired): {sorted(edges_initial_big)}")#DEGUG
                nodes, edges_repaired_big = read_ig(join(BIG_IGS_PATH, f'{case_id}_big_repaired'))

                print(f'\tStarting meta')
                trace_start = time()
                fo_best, fo_curr = gen_max, gen_max
                best_meta_edges, list_best_meta_edges = [], []
                sim_all_counter, time_out_counter, no_process_tree_counter = 0, 0, 0
                sound_solution_found = False
                temp = comb['T']
                # ----

                # TRYING TO MAKE SOUND THE INITIAL SOLUTION. IF YES, ACCEPTANCE PHASE
                if comb['meta_init'] == 0:  # la soluzione iniziale è l'ig di big riparato
                    start_edges, curr_edges = edges_repaired_big.copy(), edges_repaired_big.copy()
                else:  # 1 se la soluzione iniziale è l'ig di big non riparato
                    start_edges, curr_edges = edges_initial_big.copy(), edges_initial_big.copy()
                    #print(f"\t\tInitial SPURIOUS IG: {sorted(start_edges)}") #ALESSIA 1



                # extract admissible edges, those present are not considered
                admissible_edges = evaluate_admissible_edges(nodes, start_edges)

                # print("\t\tTrying to make the initial solution sound")

                edges, admissible_edges = repair_sound(nodes, start_edges, admissible_edges, model_dep)
                #print(f"[DEBUG] Traccia {case_id} - IG dopo repair_sound: {sorted(edges)}") #ALESSIA 2
                is_sound, i_nodes, _ = soundness(nodes, edges)
                start_edges = edges.copy()

                # SALVA IL GRAFO RIPARATO INIZIALE (PRE-METAEURISTICA)
                init_repaired_path = join(BIG_IGS_PATH, f'{case_id}_big_repaired_init')
                save_and_draw_ig(init_repaired_path, nodes, edges)
                #save_and_draw_ig(big_repaired_path_noext, nodes, edges_repaired_big)

                if is_sound:
                    print("\t\tSound IG found")
                    if fitting(nodes, edges, i_nodes):
                        sound_solution_found = True  ## ALESSIA aggiunto: se la fase di iterazione fallisce, almeno i risultati iniziali (-1) verranno salvati in trace_results (prima non accadeva)
                        try:
                            align_move = count_moves(edges, edges_aligned)
                            struct_move = count_moves(edges, start_edges) - (comb['k'] * 2)
                            accept_start_time = time()

                            log_meta, sim_all_counter, curr_edges, fo_curr, best_meta_edges, fo_best, list_best_meta_edges, time_out_counter, no_process_tree_counter, iteration_data = \
                                acceptance(-1, nodes, edges, curr_edges, struct_move, align_move, fo_curr,
                                           best_meta_edges, fo_best, list_best_meta_edges, temp, gen_max, comb['beta'],
                                           comb['gamma'])

                            accept_time = time() - accept_start_time
                            if len(iteration_data) == 0:
                                iteration_data = [None] * 8
                            it_results.loc[len(it_results)] = ([seed, case_id, comb['meta_init'], -1,
                                                                log_meta] + iteration_data + [accept_time]
                                                               + [None] * comb['k'])
                            print("\t\tAcceptance OK")
                        except Exception as e:
                            print(f"\t\tAcceptance FAIL: {e}")
                # ----

                # STARTING META
                i = 0
                while i < comb['iterations']:
                    #print(f"\t\tRepaired IG (-1): {sorted(edges)}")#ALESSIA 3
                    iteration_start = time()
                    edges.clear()
                    edges += curr_edges

                    edges, admissible_edges, move_done = local_search(nodes, edges, start_edges, admissible_edges,
                                                                      comb['k'])
                    #print(f"\t\tIG after local search {i}: {sorted(edges)}")#ALESSIA 4
                    #print(f"\t\tMove done at iteration {i}: {move_done}") #ALESSIA 4.1
                    edges, admissible_edges = repair_sound(nodes, edges, admissible_edges, model_dep)
                    #print(f"\t\tRepaired IG after LS {i}: {sorted(edges)}") #ALESSIA 5

                    "Se la traccia non fitta l'ig, la generalizzazione vale 0?????"
                    if fitting(nodes, edges, i_nodes) is False:
                        #print(f"[DEBUG] Iterazione {i} - NO FIT - IG corrente: {sorted(edges)}")#DEBUG
                        print(f"\t\t{i}) No fit, skip")
                        it_results.loc[len(it_results),
                        ['seed', 'trace_id', 'meta_init', 'iteration', 'log_meta']] = [seed, case_id,
                                                                                       comb['meta_init'],
                                                                                       i, 'no_fit']
                        i += 1
                        continue

                    sound_solution_found = True
                    align_move = count_moves(edges, edges_aligned)
                    struct_move = count_moves(edges, start_edges) - (comb['k'] * 2)
                    temp *= comb['alpha']

                    "Se la traccia fitta l'ig, calcolo la generalizzazione"
                    (log_meta, sim_all_counter, curr_edges, fo_curr, best_meta_edges, fo_best, list_best_meta_edges,
                     time_out_counter, no_process_tree_counter, iteration_data) = \
                        acceptance(i, nodes, edges, curr_edges, struct_move, align_move, fo_curr, best_meta_edges,
                                   fo_best, list_best_meta_edges, temp, gen_max, comb['beta'], comb['gamma'],
                                   time_out_counter, no_process_tree_counter, sim_all_counter)

                    iteration_time = time() - iteration_start
                    if len(iteration_data) == 0:
                        iteration_data = [None] * 8

                    it_results.loc[len(it_results)] = [seed, case_id, comb['meta_init'], i,
                                                       log_meta] + iteration_data + [iteration_time] + move_done

                    i += 1

                trace_time = time() - trace_start
                # ----

                best_align_move, best_struct_move, gen_best, log_gen_best, gen_init, log_gen_init, best_fo1, best_fo2 = None, None, None, None, None, None, 0, 0
                if not sound_solution_found:
                    print('\t\tA sound solution has never been found')
                else:
                    best_align_move = count_moves(best_meta_edges, edges_aligned)
                    best_struct_move = count_moves(best_meta_edges, start_edges) - (comb['k'] * 2)

                    gen_best, log_gen_best = generalization(nodes, best_meta_edges, gen_max)
                    gen_init, log_gen_init = generalization(nodes, start_edges, gen_max)
                    best_fo1, best_fo2, _, _ = calculate_fo(nodes, best_meta_edges, best_struct_move, best_align_move,
                                                            gen_max, comb['beta'], comb['gamma'], len(best_meta_edges))
                # ----

                align_move_bigrep = count_moves(edges_repaired_big, edges_aligned)
                struct_move_bigrep = count_moves(edges_repaired_big, start_edges) - (comb['k'] * 2)
                gen_big_rep, log_gen_big_rep = generalization(nodes, edges_repaired_big, gen_max)
                bigrep_fo1, bigrep_fo2, _, _ = calculate_fo(nodes, edges_repaired_big, struct_move_bigrep, align_move_bigrep,
                                                        gen_max, comb['beta'], comb['gamma'], len(edges_repaired_big))
                # ----

                trace_results.loc[len(trace_results)] = [seed, case_id, len(nodes), trace_time,
                                                         gen_init, log_gen_init, gen_best, log_gen_best,
                                                         best_struct_move, best_align_move, best_fo1, best_fo2, best_fo1+best_fo2,
                                                         no_process_tree_counter, sim_all_counter, len(list_best_meta_edges),
                                                         len(best_meta_edges), f'{sorted(list_best_meta_edges)}', len(edges_repaired_big),
                                                         f'{sorted(edges_repaired_big)}', gen_big_rep, struct_move_bigrep, align_move_bigrep, bigrep_fo1,
                                                         bigrep_fo2, bigrep_fo1+bigrep_fo2]

                save_and_draw_ig(
                    join(IGS_PATH, f'{seed}', string_comb_parameters, f'{case_id}_meta_{string_comb_parameters}'),
                    nodes, best_meta_edges)
                time_results.loc[len(time_results)] = [string_comb_parameters, seed, case_id, trace_time]

            comb_time = time() - comb_start_time
            print(f'\n\nCombination completed for seed: {seed} in {comb_time / 60} minutes')
            trace_results.to_csv(join(TRACE_RESULTS_PATH, f"trace_results_{string_comb_parameters}.csv"), sep=';',
                                 index=False, header=True)
            it_results.to_csv(join(ITERATION_RESULTS_PATH, f"iteration_results_{string_comb_parameters}.csv"), sep=';',
                              index=False, header=True)

        time_results.to_csv(join(TIMES_PATH, f"times_{string_comb_parameters}.csv"), sep=';', index=False, header=True)
    print(f'\n\nAll combinations done in {round((time() - initial_time) / 60, 2)} minutes')

    print('Evaluating results..')
    evaluate_results()

    print('Plotting objective functions..')
    # ci mette parecchio, al momento lo commento
    # visualize_fos()


if __name__ == '__main__':
    main()
