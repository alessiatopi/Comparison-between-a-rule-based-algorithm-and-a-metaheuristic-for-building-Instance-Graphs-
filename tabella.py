import pandas as pd
import os
import glob
from collections import Counter ###


FOLDER_PATH = "/Users/alessiatopi/Desktop/thesis/ILSM/output/Helpdesk/trace_results"  #percorso della cartella dei file csv
OUTPUT_FILE = "/Users/alessiatopi/Desktop/thesis/ILSM/output/Helpdesk/confronto_meta_vs_big.csv"

#trova tutti i file csv
csv_files = glob.glob(os.path.join(FOLDER_PATH, "trace_results_*.csv"))
#print("File trovati:", csv_files)

#inizializzo un dizionario per raccogliere i risultati
from collections import defaultdict

pivot_data = defaultdict(dict)  # {trace_id: {combinazione: 0 o 1}}

#per memorizzare per ogni trace_id le combinazioni che hanno avuto meta_wins = 1
trace_winning_configs = defaultdict(list)

reason_by_trace = defaultdict(list) ###
trace_winning_seeds = defaultdict(list) ###SEME NUOVO

for file_path in csv_files:
    df = pd.read_csv(file_path, sep=';')

    # FILTRO considero solo IG valutati con successo (process_tree) ????????
    #df = df[(df['log_meta_best'] == "process_tree")]
    df = df[df['log_meta_best'].isin(["process_tree", "simulator_ok"])]

    #estraggo il nome della combinazione dal nome del file
    combination_name = os.path.basename(file_path).replace("trace_results_", "").replace(".csv", "")
    #print(f"Elaboro: {combination_name} ({len(df)} righe)")

    #per ogni riga, si confrontano fo, gen, moves
    for _, row in df.iterrows():
        trace_id = row.get("trace_id")
        seed = row.get("seed") ##SEME NUOVO

        try:
            #estraggo le metriche principali
            time_meta = float(row.get("time_meta", 0)) ##TEMPO

            fo_meta = float(row.get("fo_meta_best", 0))
            fo_big = float(row.get("fo_big_rep", 0))

            gen_meta = float(row.get("gen_meta_best", 0))
            gen_big = float(row.get("gen_big_rep", 0))

            struct_move_meta = int(row.get("best_meta_struct_move", 0))
            struct_move_big = int(row.get("big_rep_struct_move", 0))

            align_move_meta = int(row.get("best_meta_align_move", 0))
            align_move_big = int(row.get("big_rep_align_move", 0))

            #edges_best_meta = str(row.get("edges_best_meta", ""))
            #edges_best_big = str(row.get("edges_big_rep", ""))
            edges_meta = str(row.get("num_edges_best_meta", 0))
            edges_big = str(row.get("num_edges_big_rep", 0))

            # logica di confronto
            '''meta_wins = 0
            if (fo_meta < fo_big) and (gen_meta < gen_big):
                meta_wins = 1
                reason = "FO_and_GEN"
                # print(f"{trace_id} → META vince: FO({fo_meta}<{fo_big}), GEN({gen_meta}<{gen_big})")
                print(
                    f"{trace_id} (seed {seed}) → [{combination_name}] META vince: FO({fo_meta}<{fo_big}), GEN({gen_meta}<{gen_big}), STRUCT({struct_move_meta}>{struct_move_big})")
            pivot_data[trace_id][combination_name] = meta_wins'''

            # logica di confronto
            meta_wins = 0
            if (fo_meta < fo_big) and struct_move_meta != -2 and struct_move_meta != -4:
                meta_wins = 1
                reason = "FO_only"
                # print(f"{trace_id} → META vince: FO({fo_meta}<{fo_big}), GEN({gen_meta}<{gen_big})")
                print(
                    f"{trace_id} (seed {seed}) → [{combination_name}] META vince: FO({fo_meta}<{fo_big}), GEN_META({gen_meta}vs{gen_big}), STRUC_META({struct_move_meta}vs{struct_move_big})")
            pivot_data[trace_id][combination_name] = meta_wins


            if meta_wins == 1:
                trace_winning_configs[trace_id].append(combination_name)
                reason_by_trace[trace_id].append(reason) ###
                trace_winning_seeds[trace_id].append((combination_name, seed)) ###SEME nuovo
                print(f"{trace_id} (seed {seed}) → META ha vinto in {time_meta:.2f}s") #TEMPO

        except Exception as e:
            print(f"Errore su trace_id {trace_id} in {combination_name}: {e}")
            pivot_data[trace_id][combination_name] = None

#creazione di un dataframe pivot
pivot_df = pd.DataFrame.from_dict(pivot_data, orient='index')
pivot_df.index.name = "trace_id"

#salvataggio del csv
pivot_df.to_csv(OUTPUT_FILE, sep=';')
#print(f"File pivot salvato in: {OUTPUT_FILE}")

#conto le righe dove almeno un valore è 1 (quindi la metaeuristica ha vinto almeno una volta per quella traccia)
at_least_one_win = pivot_df.eq(1).any(axis=1)

#estaraggo i trace_id corrispondenti
winning_trace_ids = pivot_df.index[at_least_one_win].tolist()

#conto quanti sono
num_winning_traces = len(winning_trace_ids)

print(f"Numero di tracce in cui la metaeuristica ha vinto almeno una volta: {num_winning_traces}")
print("Tracce IDs corrispondenti:")
print(winning_trace_ids)

print("\nConfigurazioni vincenti per ogni traccia_id:")
for trace_id in sorted(trace_winning_configs):
    configs = trace_winning_configs[trace_id]
    seeds_info = trace_winning_seeds[trace_id] ###SEME NUOVO
    #print(f"{trace_id} → Configs: {configs}, Seeds: {seeds_info}")
    print(f"{trace_id} : {configs}")
    #print(f"{trace_id} → {configs}")


###
print("\n--- Riassunto vittorie META per traccia ---")
for trace_id, configs in trace_winning_configs.items():
    reasons = reason_by_trace[trace_id]
    reason_counts = Counter(reasons)
    reason_most_common = reason_counts.most_common(1)[0][0] if reason_counts else "N/A"
    #print(f"Trace {trace_id}: META ha vinto in {len(configs)} combinazioni, migliorando per: {reason_most_common}")
    # estrai i seed diversi per questa trace_id
    seeds_info = trace_winning_seeds[trace_id]
    seeds = sorted(set([seed for (_, seed) in seeds_info]))  # prendi solo i seed, rimuovi duplicati
    print(f"Trace {trace_id}: META ha vinto in {len(configs)} combinazioni, migliorando per: {reason_most_common}, Seeds: {seeds}")

###integra il fatto di vedere per ogni combinazione che risultano vincenti, il numero di tracce
###oppure combinazioni/confogurazioni più comuni dentro le liste delle tracce

# Crea dizionario inverso a partire da trace_winning_configs
winning_traces_by_combination = defaultdict(list)

for trace_id, combinations in trace_winning_configs.items():
    for combination in combinations:
        winning_traces_by_combination[combination].append(trace_id)

# Stampa le combinazioni vincenti e le tracce corrispondenti
'''print("\n--- Riepilogo combinazioni vincenti ---")
for combination, traces in winning_traces_by_combination.items():
    unique_traces = sorted(set(traces))  # rimuove duplicati e ordina
    print(f"Combinazione '{combination}' ha vinto per le tracce: {unique_traces}")'''

# Trova le tracce escluse (mai vincitrici)
excluded_trace_ids = pivot_df.index[~at_least_one_win].tolist()

