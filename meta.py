from time import time
import sys
import math
import random
from collections import Counter
from pm4py.objects.petri_net.utils import petri_utils
from pm4py.algo.simulation.playout.process_tree import algorithm as tree_playout
from pm4py.objects.conversion.wf_net import converter as wf_net_converter
from pm4py.objects import petri_net
from pm4py.objects.petri_net.obj import PetriNet, Marking
from config import MAX_TRACE_LENGTH, LIMIT_TIME_SIMULATOR
from big import find_dependency, final_restore, soundness


def create_petri_net(nodes, edges):
    net = PetriNet("Instant Graph")
    # build a transaction for each node
    tran = []
    i = 0
    # per ogni nodo dell'ig aggiunge alla lista tran un oggetto di tipo Transation(<nome>, <etichetta>)
    # e aggiunge la transizione alla rete di petri
    for n in nodes:
        tran.append(PetriNet.Transition(n, n))
        net.transitions.add(tran[i])
        i += 1
    # for every arc build a place and two arcs
    # build also source and sink
    places = []
    i = 0
    for a in edges:
        name = "p_" + str(i + 1)
        places.append(PetriNet.Place(name))
        net.places.add(places[i])
        a1 = nodes.index(a[0])
        a2 = nodes.index(a[1])
        petri_utils.add_arc_from_to(tran[a1], places[i], net)
        petri_utils.add_arc_from_to(places[i], tran[a2], net)
        i += 1
    source = PetriNet.Place("source")
    sink = PetriNet.Place("sink")
    net.places.add(source)
    net.places.add(sink)
    petri_utils.add_arc_from_to(source, tran[0], net)
    petri_utils.add_arc_from_to(tran[len(tran) - 1], sink, net)
    # names initial and final marking
    im = Marking()
    im[source] = 1
    fm = Marking()
    fm[sink] = 1
    return net, im, fm


def calculate_fo(V, W, struct_move, align_move, gen_max, beta, gamma, m):
    gen1, log = generalization(V, W, gen_max)
    if log == "time_out":
        return 0, 0, log, 0

    # ----- normalizing values ------
    if struct_move < 0:
        struct_move = 0
    else:
        struct_move = struct_move / (m * 2)

    if align_move < 0:
        align_move = 0
    else:
        align_move = align_move / (m * 2)

    gen = gen1 / gen_max

    fo1 = beta * gen
    fo2 = (1 - beta) * (gamma * struct_move + (1 - gamma) * align_move)
    return fo1, fo2, log, gen1


# function that holds or discards igs based on the result of the objective function
def acceptance(i, V, W, curr_e, struct_move, align_move, fo_curr, best_meta_edges, fo_best,
               list_best_meta_edges, T, gen_max, beta, gamma, time_out_counter=0, no_process_tree_counter=0,
               sim_all_counter=0):
    # calcolo generalizzazione
    fo1, fo2, d, gen = calculate_fo(V, W, struct_move, align_move, gen_max, beta, gamma, len(curr_e))

    if d == "time_out":
        time_out_counter += 1
        no_process_tree_counter += 1
        return d, sim_all_counter, curr_e, fo_curr, best_meta_edges, fo_best, list_best_meta_edges, time_out_counter, no_process_tree_counter, []

    elif d == "no_process_tree":
        no_process_tree_counter += 1

    fo = fo1 + fo2

    ris = 0  # reject the solution
    result = ''
    if fo == fo_curr:
        ris = random.randint(0, 1)
        result = 'new one rejected (equals)'
    if ris:
        ris = 2  # take the new one
        result = 'new one accepted (equals)'
    if fo > fo_curr:
        ris, prob, z = simulated_annealing(fo_curr, fo, T)
        result = 'new one rejected'
        if ris:  # ris = 1 take from sim annealing
            sim_all_counter += 1  # incrementa il numero di volte che è stata accettata una soluzione peggiorativa dal
            # simulation annealing
            result = 'new one accepted (simulated annealing)'
    if fo < fo_curr:
        ris = 3  # the new one is better
        result = 'new one accepted'

    fo_prev = fo_curr
    if i == -1:
        fo_prev = None
    # se ris > 0 allora si accetta la soluzione nuova
    if ris > 0:
        # update IG CURR values
        curr_e.clear()
        curr_e += W
        fo_curr = fo
        if fo < fo_best:
            ris = 4  # the new one is the best
            fo_best = fo
            list_best_meta_edges.clear()
            print(f'Best found at iteration {i}')
            list_best_meta_edges.append((V, W))
            best_meta_edges.clear()
            best_meta_edges += W
            result = 'new one accepted (new best)'

    check_eq = False
    if fo == fo_best:
        for (nod, edg) in list_best_meta_edges:  # scroll all the igs best
            for e in W:  # for all the edge in the current ig check if is in the ig best in the list
                if e not in edg:
                    check_eq = True
                    break
            if check_eq:
                break
        if check_eq:
            list_best_meta_edges.append((V, W))

    if i == -1:
        fo_prev = fo

    list = [fo_prev, fo, fo1, fo2, result, gen, struct_move, align_move]  # data to save in csv file
    print(f'\t\t{i}) fo: {fo}, result: {result}')

    return d, sim_all_counter, curr_e, fo_curr, best_meta_edges, fo_best, list_best_meta_edges, time_out_counter, no_process_tree_counter, list


# return 1 if take IG' like new IG, 0 otherwise
def simulated_annealing(fo_curr, fo1, T):
    value = 0 if T == 0 else round((fo_curr - fo1) / T, 5)
    prob = math.exp(value)
    z = random.random()
    if z < prob:
        return 1, prob, z
    else:
        return 0, prob, z


# count the different arcs between the two igs considered
def count_moves(curr, start):
    count = 0
    for e in start:
        if e not in curr:
            count += 1  # count remove
    for e in curr:
        if e not in start:
            count += 1  # count add
    return count


def simulator_apply(net, im, fm, gen_max, max_trace_len):
    max_marking_occ = sys.maxsize
    semantics = petri_net.semantics.ClassicSemantics()

    feasible_elements = []

    to_visit = [(im, (), ())]
    visited = set()
    t_init = time()
    while len(to_visit) > 0:
        t_curr = time() - t_init
        # ***** TIMEOUT *****
        if t_curr > LIMIT_TIME_SIMULATOR:
            # ritorna la generalizzazione calcolata fino a quel momento
            out = gen_max if len(feasible_elements) == 0 else len(feasible_elements)
            return out, 'timeout'

        state = to_visit.pop(0)

        m = state[0]
        trace = state[1]
        elements = state[2]

        try:
            if trace in visited:
                continue
            visited.add((m, trace))
        except Exception as e:
            print("Failed simulator_apply:", e)
            # se fallisce, nel dubbio riporto il valore massimo
            return gen_max, 'failed_simulator'

        en_t = semantics.enabled_transitions(net, m)

        if (fm is not None and m == fm) or (fm is None and len(en_t) == 0):
            if len(trace) <= max_trace_len:
                feasible_elements.append(elements)

        for t in en_t:
            new_elements = elements + (m,)
            new_elements = new_elements + (t,)

            counter_elements = Counter(new_elements)

            if counter_elements[m] > max_marking_occ:
                continue

            new_m = semantics.weak_execute(t, net, m)
            if t.label is not None:
                new_trace = trace + (t.label,)
            else:
                new_trace = trace

            new_state = (new_m, new_trace, new_elements)

            if new_state in visited or len(new_trace) > max_trace_len:
                continue
            to_visit.append(new_state)
    # tutto a buon fine, riporto il valore calcolato
    return len(feasible_elements), 'simulator_ok'


def generalization(nodes, edges, gen_max):
    # si crea la petri net dell'attuale IG
    new_nodes = []
    new_edges = []
    for n in nodes:
        new_nodes.append(str(n[0]) + '_' + n[1])
    for e in edges:
        new_edges.append((str(e[0][0]) + '_' + e[0][1], str(e[1][0]) + '_' + e[1][1]))
    net, im, fm = create_petri_net(new_nodes, new_edges)

    try:
        "Generalizzazione calcolata correttamente"
        # si cerca di convertire la petri net derivata dall'ig in una wf net
        tree = wf_net_converter.apply(net, im, fm)
        param = tree_playout.Variants.EXTENSIVE.value.Parameters
        simulated_log = tree_playout.apply(tree, variant=tree_playout.Variants.EXTENSIVE,
                                           parameters={param.MAX_TRACE_LENGTH: MAX_TRACE_LENGTH,
                                                       param.MAX_LIMIT_NUM_TRACES: gen_max})
        gen = len(simulated_log)
        return gen, 'process_tree'

    except Exception as e:
        if e.args[0] == "The Petri net provided is not a WF-net":
            # save_graph("not_wf_net.g", nodes, edges)
            return gen_max, 'no_wf_net'
        """
        Provo a fare simulazione direttamente sulla petri net, ho tre casi
        1) vado in timeout per il troppo tempo impiegato, in questo caso gen = 0???????, timeout = 1
        2) la simulazione fallisce, in questo caso gen = 0, timeout = 1 (dovrebbe essere così per timeout)
        3) la simulazione va a buon fine, in questo caso gen = g, timeout = 0
        """
        gen, status = simulator_apply(net, im, fm, gen_max, len(nodes))
        return gen, status


def fitting(V, W, inode):
    # V -> list of nodes sorted by order of execution
    p = []  ### nodi già visitati
    m = [] ### vicini da esplorare
    n = [inode] ### nodi esplorabili in questo momento
    for ev in V:
        if ev in n:
            n.remove(ev)
            p.append(ev)
            for arc in W:
                if arc[0] == ev and arc[1] not in m:
                    m.append(arc[1])
            n += m
            for p1 in p:
                if p1 in n:
                    n.remove(p1)
        else:
            # print('\t\tNO FITTING')
            return False
    return True


def get_parent_nodes_from_graph(nodes, edges):
    parent_elements = {node: [] for node in nodes}
    for node in nodes:
        parents = [start for start, end in edges if end == node]
        if len(parents) != 0:
            parent_elements[node].extend(parents)

    return parent_elements


def check_transitivity(node_start, node_end, parent_nodes):
    # caso base 1: node_end e node_start sono paralleli, quindi l'arco non è transitivo
    if parent_nodes[node_end] == parent_nodes[node_start]:
        return False

    # caso base 2: node_start è padre di node_end, quindi l'arco è transitivo
    if node_start in parent_nodes[node_end]:
        return True

    # ricorsione: node_start non è padre diretto di node_end, devo vedere se lo è per i nodi intermedi, se sì l'arco è transitivo
    if node_start not in parent_nodes[node_end]:
        for mid_node in parent_nodes[node_end]:
            if check_transitivity(node_start, mid_node, parent_nodes):
                return True

    return False


# extract only admissible edges not present in the current IG
def evaluate_admissible_edges(nodes, edges): #modifica alessia
    admissible_edges = []
    for i in range(0, len(nodes)):
        for f in range(i + 1, len(nodes)):
            edge = (nodes[i], nodes[f])
            edge_not_exists = edge not in edges
            if edge_not_exists:
                # Costruisci parent_nodes senza l'arco corrente
                edges_without_edge = edges.copy()
                parent_nodes = get_parent_nodes_from_graph(nodes, edges_without_edge)
                edge_is_transitive = check_transitivity(nodes[i], nodes[f], parent_nodes)
                if not edge_is_transitive:
                    admissible_edges.append(edge)
    return admissible_edges


def repair_sound(V, W, ad_e, m_dep): #moficato alessia
    check_s, inodes, fnodes = soundness(V, W)
    if check_s is False:
        W, ad_e = add_edge_nosound(inodes, fnodes, V, W, ad_e, m_dep)
    # 📌 Filtro archi transitivi
    W_copy = W.copy()
    for edge in W_copy:
        start, end = edge
        edges_without_edge = [e for e in W if e != edge]
        parent_nodes = get_parent_nodes_from_graph(V, edges_without_edge)
        if check_transitivity(start, end, parent_nodes):
            W.remove(edge)
    return W, ad_e ##grafo con eventuali archi nuovi, rimanenti archi ammissibili


# add edges to make the ig sound
def add_edge_nosound(inodes, fnodes, V, W, ad_e, dependency):
    # nodes (activity) are ordered by execution
    for node_target in inodes:
        old = []
        i = V.index(node_target)
        if i == 0:
            continue
        while True:
            node_source = V[i - 1]
            edge = (node_source, node_target)
            if edge in ad_e and node_source[1] in dependency[node_target[1]]:
                W.append(edge)
                ad_e.remove(edge)
                if edge[0] in fnodes:
                    fnodes.remove(edge[0])
                break
            else:
                i -= 1
                old.append(edge)
                if i < 1:
                    for edge in old:
                        if edge in ad_e:
                            W.append(edge)
                            ad_e.remove(edge)
                            if edge[0] in fnodes:
                                fnodes.remove(edge[0])
                            break
                    break
                continue
    for node_source in fnodes:
        old = []
        i = V.index(node_source)
        if i == len(V) - 1:
            continue
        while True:
            node_target = V[i + 1]
            edge = (node_source, node_target)
            if edge in ad_e and node_source[1] in dependency[node_target[1]]:
                W.append(edge)
                ad_e.remove(edge)
                break
            else:
                i += 1
                old.append(edge)
                if i == len(V) - 1:
                    for edge in old:
                        if edge in ad_e:
                            W.append(edge)
                            ad_e.remove(edge)
                            break
                    break
                continue
    return W, ad_e


# updates the dependencies each time I remove an arc and restore the path
def local_search(V, W, iW, ad_e, k):
    old = []
    rnd = random.randint(0, len(W))
    move_done = []
    if rnd == len(W):
        a = "No Edge"
    else:
        a = W[rnd]

    while k:  # chose k arcs to remove
        k -= 1
        while a in old:  # this arc must not already be chosen
            rnd = random.randint(0, len(W))
            if rnd == len(W):
                a = "No Edge"
            else:
                a = W[rnd]
        old.append(a)

        W, ad_e = remove_edge(a, W, ad_e)
        # find the path to restore
        if a in iW:
            redges = restore_path(a[0], a[1], W, ad_e)
            if redges[0] == a:
                W, ad_e = add_edge(a, W, ad_e)
                move_done.append("No Path")
            else:
                cont = 0
                for red in redges:
                    if red not in W:
                        W, ad_e = add_edge(red, W, ad_e)  # IMPO UPDATE MARZO 2022
                        old.append(red)
                        cont += 1
                move_done.append("Restore " + str(cont) + " edges")
        elif a != "No Edge":
            move_done.append("Remove edge")
        else:
            move_done.append("No Edge")
        dep = find_dependency(V, W)
        W = final_restore(V, W, dep)
    return W, ad_e, move_done ###restituisce il nuovo grafo, i nuovi archi ammissibili e la lista delle mosse effettuate


# inode e fnode sono i nodi dell'arco rimosso
def restore_path(inode, fnode, W, ad_e):
    redges = []
    n1 = inode
    while True:
        arcs = []
        for e in ad_e:
            if e[0] == n1 and e[1][0] <= fnode[0]:
                # take the arcs with n1 == inode if they have for n2 a node with id <= id_fnode
                arcs.append(e)
        # chose random arc from those available
        # if arcs == [] means that arc already exist to restore the path
        if not arcs:
            for e in W:
                if e[0] == n1 and e[1][0] <= fnode[0]:
                    # chose path considered existing arc
                    arcs.append(e)
        if not arcs:
            print("\t\tCancel, stop in the rebuilding of path")
            redges.append(
                (inode, fnode))  # ripristina l'arco rimosso perchè non c'è altro modo di ripristinare il percorso
            break
        if n1 == inode and len(arcs) > 1:
            rnd_id = random.randint(0,
                                    len(arcs) - 2)  # last element is the arc removed before, the last to be added to ad_e
        else:
            rnd_id = random.randint(0, len(arcs) - 1)  # chose the last if there is only one
        edge = arcs[rnd_id]
        redges.append(edge)
        if edge[1] == fnode:  # not exist alternative path
            break
        else:
            n1 = edge[1]

    return redges


def add_edge(edge, W, ad_e):
    W.append(edge)
    ad_e.remove(edge)
    return W, ad_e ###restituisce le due liste aggiornate


def remove_edge(edge, W, ad_e):
    if edge != "No Edge":
        ad_e.append(edge)
        W.remove(edge)
    return W, ad_e
