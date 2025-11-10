from os.path import join, abspath, dirname, exists
from os import makedirs
BASE_PATH = dirname((abspath(__file__)))


LOG_NAME = 'Helpdesk'
#LOG_NAME = 'PrepaidTravelCosts'

# combinazioni
ALPHA = [0.3]
BETA = [0.5]
GAMMA = [0]  # GAMMA = 0 e BETA = 0.75 -> best values by experiments in gap gen and gap move terms
T = [500]
ITERATIONS = [20]
META_INIT = [ 1]  # 0 ='BIG_REPAIRED', 1 = 'BIG_NOT_REPAIRED'
K = [1]  # aumentare, provando 2-(4)-6-(8)-10 (4 e 8 per ora no)
SEEDS = [1]

# costanti della metaeuristica
MAX_LIMIT_NUM_TRACES = 100_000  # threshold per valore massimo durante la generazione sul process tree
LIMIT_TIME_SIMULATOR = 10  # tempo massimo della procedura di simulazione in secondi
MAX_TRACE_LENGTH = 1_000  # numero massimo di eventi possibili all'interno di una traccia

# paths
DATASET_PATH = join(BASE_PATH, 'dataset')
OUTPUT_PATH = join(BASE_PATH, 'output', LOG_NAME)

NET_FILE_PATH = join(DATASET_PATH, f"{LOG_NAME}_net.pnml")
LOG_FILE_PATH = join(DATASET_PATH, f"{LOG_NAME}.xes")

TRACE_RESULTS_PATH = join(OUTPUT_PATH, "trace_results")
ITERATION_RESULTS_PATH = join(OUTPUT_PATH, "iteration_results")
TIMES_PATH = join(OUTPUT_PATH, "times")
IGS_PATH = join(OUTPUT_PATH, 'igs')
PLOTS_PATH = join(OUTPUT_PATH, 'plots')
TRACES_MIN_FO_PATH = join(OUTPUT_PATH, 'traces_min_fo_iteration')
ALIGNED_IGS_PATH = join(IGS_PATH, 'aligned')
BIG_IGS_PATH = join(IGS_PATH, 'big')

ALIGNED_CSV_FILE_PATH = join(OUTPUT_PATH, f"aligned_igs.csv")
BIG_TIMES_FILE_PATH = join(TIMES_PATH, f'times_big.csv')


# combinazioni da testare
COMBINATIONS = []
for meta_init in META_INIT:
    for alpha in ALPHA:
        for beta in BETA:
            for gamma in GAMMA:
                for t in T:
                    for k in K:
                        for iterations in ITERATIONS:
                            COMBINATIONS.append({
                                'meta_init': meta_init,
                                'alpha': alpha,
                                'beta': beta,
                                'gamma': gamma,
                                'T': t,
                                'k': k,
                                'iterations': iterations
                            })


# crea cartelle per non andare in errore
for path in [TRACE_RESULTS_PATH, ITERATION_RESULTS_PATH, TIMES_PATH, IGS_PATH, PLOTS_PATH ,ALIGNED_IGS_PATH,
             BIG_IGS_PATH, TRACES_MIN_FO_PATH]:
    if not exists(path):
        makedirs(path, exist_ok=True)
for comb in COMBINATIONS:
    for seed in SEEDS:
        t, iterations, alpha, k, beta, gamma, meta_init = (comb['T'], comb['iterations'], comb['alpha'],
                                                          comb['k'], comb['beta'], comb['gamma'], comb['meta_init'])
        comb_parameters = f"a{alpha}_b{beta}_g{gamma}_T{t}_it{iterations}_k{k}_mi{meta_init}"
        comb_path = join(IGS_PATH, f'{seed}', comb_parameters)

        if not exists(comb_path):
            makedirs(comb_path, exist_ok=True)


def get_comb_parameters(comb):
    T, iterations, alpha, k, beta, gamma, meta_init = (comb['T'], comb['iterations'], comb['alpha'],
                                                       comb['k'], comb['beta'], comb['gamma'], comb['meta_init'])
    comb_parameters = f"a{alpha}_b{beta}_g{gamma}_T{T}_it{iterations}_k{k}_mi{meta_init}"
    return comb_parameters
