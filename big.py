import pm4py
from pm4py.streaming.importer.xes import importer as xes_importer
from time import time
from pm4py.objects.petri_net.importer import importer as pnml_importer
from config import BIG_IGS_PATH, BIG_TIMES_FILE_PATH
from utils import save_and_draw_ig, write_or_add_text_to_file
import pandas as pd
from os.path import join, exists
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
from ast import literal_eval
from config import ALIGNED_CSV_FILE_PATH, ALIGNED_IGS_PATH
import pdb

# usare soundness di pm4py

def find_successors(net):
    return {transition: find_successors_of_transition(net, transition) for transition in net.transitions if transition.label is not None}


def find_causal_relationships(net):
    dict_succ = find_successors(net)
    result = []
    for key, item in dict_succ.items():
        for s in item:
            result.append((key.label, s.label))
    return result


def find_successors_of_transition(net, transition):
    sources = {transition}
    targets = set()
    visited = set()
    while sources:
        source = sources.pop()
        if not (type(source) is pm4py.objects.petri_net.obj.PetriNet.Transition and source.label is not None):
            visited.add(source)
        for arc in source.out_arcs:
            if arc.target in visited:
                continue
            if type(arc.target) is pm4py.objects.petri_net.obj.PetriNet.Transition and arc.target.label is not None:
                targets.add(arc.target)
            else:
                sources.add(arc.target)
    return targets


def pick_aligned_trace(trace, net, initial_marking, final_marking):
    aligned_traces = alignments.apply_trace(trace, net, initial_marking, final_marking)
    #print("alignment")
    #print(aligned_traces['alignment'])
    temp = []
    id = 0
    al = []
    temp1=[]
    id1=0
    fin = []
    for edge in aligned_traces['alignment']:
        id+=1
        temp.append((id,edge[1]))
    al.append(temp)
    for edge in aligned_traces['alignment']:
        id1+=1
        temp1.append((id1,edge[0]))
    fin.append(temp1)

    return al, fin


def check_trace_conformance(trace, net, initial_marking, final_marking):
    aligned_traces = alignments.apply_trace(trace, net, initial_marking, final_marking)
    D = []
    I = []
    id = 0
    temp_d = []
    temp_i = []
    prev_d = False
    curr_d = False
    prev_i = False
    curr_i = False
    del_count = 1
    for edge in aligned_traces['alignment']:
        pdb.set_trace()
        id+=1
        if edge[1] is None:
            id-=1
            continue
        if edge[0] == '>>':
            temp_d.append((id, edge[1]))
            curr_d = True
            id-=1
        if edge[1] == '>>':
            temp_i.append((id, edge[0]))
            curr_i = True

        if (prev_i and not curr_i):
            if len(temp_i) > 0:
                I.append(temp_i)
            temp_i = []
        prev_i = curr_i
        curr_i = False
        if (prev_d and not curr_d):
            if len(temp_d) > 0:
                D.append(temp_d)
            temp_d = []

        prev_d = curr_d
        curr_d = False
    if len(temp_i) > 0:
        I.append(temp_i)
    if len(temp_d) > 0:
        D.append(temp_d)
    return D, I



def mapping(L1,L2):
    map = [0]*len(L1)
    id1 = 0
    id2 = 0
    ins = []
    for i in range(len(L1)):
        e1=L1[i]
        e2=L2[i]
        if e1[1]==e2[1]:
            id1+=1
            id2+=1
            map[i] = (e1[1],id1, id2 )
        elif e1[1]=='>>': #insertion
            id1+=1
            map[i]= (e2[1],id1, 0)
        elif e2[1]=='>>': #deletion
            #pdb.set_trace()
            id2+=1
            map[i]=(e1[1],0,id2)


    for j in range(len(L1)):
        e1 = L1[j]
        e3 = map[j]
        if e1[1]=='>>':
            id2+=1
            map[j] = (e3[0], e3[1], id2)
            ins.append((e3[0], e3[1], id2))


    return map, ins


def extract_instance_graph(trace, cr):
    V = []
    W = []
    id = 1
    for event in trace:
      #V.append((id, event.get("concept:name")))
      V.append(event)
      id += 1
    # print("IG")
    for i in range(len(V)):
      for k in range(i+1,len(V)):
        e1 = V[i]
        e2 = V[k]
        if (e1[1],e2[1]) in cr:
          flag_e1=True
          for s in range(i+1, k):
            e3 = V[s]
            if (e1[1],e3[1]) in cr:
              flag_e1 = False
              break
          flag_e2=True
          for s in range(i+1, k):
            e3 = V[s]
            if (e3[1],e2[1]) in cr:
              flag_e2 = False
              break

          if flag_e1 or flag_e2:
            W.append((e1,e2))
    return V, W


def compliant_trace(trace):
  t = []
  id = 0
  for event in trace:
    if event[1] == '>>':
      continue
    else:
      id +=1
      t.append((id,event[1]))

  return t


def del_repair(V, W ,map ,deletion):
    Eremp = []
    Erems = []
    Pred = []
    Succ = []
    W1 = []
    V1 = []
    d = []
    W2 = []


    to_del = (deletion[2],deletion[0])

    for i in range(len(W)):
        e1 = W[i]
        e2 = e1[1]
        e3 = e1[0]
        if e2 == to_del:
            Eremp.append((e3,to_del))
        if e3 ==to_del:
            Erems.append((to_del,e2))

    for a in Eremp:   #crea liste Pred e Succ
        Pred.append(a[0])
    for b in Erems:
        Succ.append(b[1])

    for ep in Eremp:
        W.remove(ep)
    for es in Erems:
        W.remove(es)

    V.remove(to_del)

    for p in Pred:
        for s in Succ:
            W.append((p,s))


    return V,W


def ins_repair(V, W, map, insertion, V_n, ins_list, Vpos):
   Eremp = []
   Pred = []
   Succ = []
   pos_t = []
   W_num = []
   V1 = []
  # ins_num = []
#


   V.insert(insertion[1]-1,(insertion[2],insertion[0]))
   Vpos.insert(insertion[1]-1, (insertion[2], insertion[0]))
   pos_t.append(insertion[1])

   W_num = edge_number(W)
   V_num = node_number(V)
  # ins_num = ins_list_num(ins_list)

   #print('ins num: ',ins_num)

  # print('Vpos agg: ', Vpos)

   for p in pos_t: # numero dell'insertion
       #print('P=',p)
       #print('Len Vpos: ', len(Vpos))
       if p < len(Vpos):
         position = Vpos[p] #posizione in cui va inserito il nodo
        # print('Position = ', position)
         pos = position[0]
       else:   #inserimento a ultimo posto. la posizione di inserimento è maggiore o uguale della lunghezza di Vpos (che non considera nodi da cancellare)
         position = V[-1]
         pos = position[0]
         #print('ULtimo elemento ', Vpos[-2])
         #print(pos)

       # print('pos: ', pos)
       p_pred = Vpos[p-2] #in Vpos il nodo da inserire viene messo in posizione p-1 perchè il vettore parte da 0, quindi il precedente lo trovo come p-2
       pos_pred = p_pred[0]
      # print('P pred: ', p_pred)
       #linee 6-12 pseudocodice
       if is_path(pos_pred,pos,W_num,V):  # se c'è un cammino tra p-1 e p
         # print(is_path(pos_pred,pos,W_num,V))
         for i in range(len(W)):
           arc = W[i]
           a0 = arc[0]
           a1 = arc[1]
           if pos == a1[0] and (a0,a1) not in Eremp:
             # se esiste un arco nel grafo che entra in posizione p e non è ancora stato inserito in Eremp
             # si trovano gli archi entranti (e quindi i nodi Pred) nel nodo in posizione in cui va fatto l'inserimento
             Eremp.append((a0,a1))
             Pred.append(a0)
         for n in Pred:        #linee 9-10 pseudocodice, si controllano eventuali parallelismi non considerati nel ciclo precedente
           for k in range(len(W)):
             e = W[k]
             e0 = e[0]
             e1 = e[1]
             if e0 == n and (e0,e1) not in Eremp:
               Eremp.append((e0,e1))
       else: #linee 14-15 pseudocodice, l'insertion avviene all'interno di un parallelismo
         for m in range(len(W)):
           edge = W[m]
           edge0 = edge[0]
           edge1 = edge[1]
           if (pos_pred) == edge0[0] and (edge0,edge1) not in Eremp:
             Eremp.append((edge0,edge1))
             Pred.append(edge0)
           elif (pos_pred) == edge1[0] and (pos_pred) == V_n[-1]:  #insertion all'ultimo nodo del grafo
             Pred.append(edge1)




   #linea 17 pseudocodice
   for erem in range(len(Eremp)):
     suc = Eremp[erem]
     suc1 = suc[1]
     if suc1 not in Succ:
       Succ.append(suc1)

   #print('Pred = ', Pred)
   #print('Succ = ', Succ)
   #print('Eremp = ', Eremp)

   #linea 18 pseudocodice
   for el in Eremp:
     if el in W:
       W.remove(el)


   for i in Pred:
     if (i,(insertion[2],insertion[0])) not in W:
       W.append((i,(insertion[2],insertion[0])))

   for s in Succ:
     if ((insertion[2],insertion[0]),s) not in W:
       W.append(((insertion[2],insertion[0]),s))


   W_num = edge_number(W)
   V_num = node_number(V)

  # print('V: ', V)
   #print('VPOS Finale: ', Vpos)

   #print('++++++++++++++')


   return V,W


def edge_number(W):

  W_number = []

  for i in range(len(W)):
    arc = W[i]
    a0 = arc[0]
    a1 = arc[1]
    W_number.append((a0[0],a1[0]))

  return W_number


def node_number(V):

  V_number = []
  for i in range(len(V)):
    nod = V[i]
    V_number.append(nod[0])

  return V_number


def is_path(a, b, W, V):
  flag = False
  if (a,b) in W:
    flag = True
    return flag
  else:
    for c in range(len(V)):
      e = V[c]
      if (a,e[0]) in W:
        flag = is_path(e[0],b,W,V)
      else:
        continue

  return flag


def update_label(W, map, V):

  W1 = []
  V1 = []


  for i in range(len(W)):
    arc = W[i]
    a0 = arc[0]
    a1 = arc[1]
    for j in range(len(map)):
      e = map[j]
      if a0 == (e[2],e[0]):
        for k in range(len(map)):
          f = map[k]
          if a1 == (f[2],f[0]):
            W1.append(((e[1],e[0]),(f[1],f[0])))


  for i1 in range(len(V)):
    node = V[i1]
    for j1 in range(len(map)):
      e = map[j1]
      if node == (e[2],e[0]):
        V1.append((e[1],e[0]))


  return W1, V1


def save_g_file(V, W, path, time, sort_labels):
    with open(path, 'w') as f:
        # f.write("# Execution Time: {0:.3f} s\n".format(time))
        # f.write("# Deleted Activities: {0}\n".format(D))
        # f.write("# Inserted Activities: {0}\n".format(I))
        for n in V:
            f.write("v {0} {1}\n".format(n[0], n[1]))
        f.write("\n")
        if (sort_labels):
            W.sort()
        for e in W:
            f.write("e {0} {1} {2}__{3}\n".format(e[0][0], e[1][0], e[0][1], e[1][1]))


def BIG(net_path, log_path, tr_start=0, tr_end=None, sort_labels=False):
    streaming_ev_object = xes_importer.apply(log_path, variant=xes_importer.Variants.XES_TRACE_STREAM)  # file xes
    net, initial_marking, final_marking = pnml_importer.apply(net_path)
    cr = find_causal_relationships(net, initial_marking, final_marking)

    n = 0
    for trace in streaming_ev_object:
        n += 1
        Aligned, A = pick_aligned_trace(trace, net, initial_marking, final_marking)
        Align = Aligned[0]
        A1 = A[0]
        map, ins = mapping(Align, A1)

        # to generate the IG on the model
        compliant = compliant_trace(Align)
        # effettiva = compliant_trace(A1)

        d = []

        trace_start_time = time()
        #estrazione dell' IG su cui poi devo fare riparazione
        V, W = extract_instance_graph(compliant, cr)

        V_n = node_number(V)
        W_n = edge_number(W)

        for element in map:  # crea lista dei nodi da cancellare
            if element[1] == 0:
                d.append(element)

        # Vpos lista dei nodi utilizzata per repair. inizialmente si rimuovono da essa i nodi relativi a deletion.
        # In seguito viene passata in input alla funzione di insertion repair e ogni attività viene inserita all'interno di Vpos
        # nella posizione di inserimento. così facendo vpos sarà sempre aggiornata ad ogni inserimento.
        # al termine delle insertion avrò la mia lista dei nodi aggiornata.

        Vpos = []
        for node in V:
            Vpos.append(node)

        for el in map:
            if el[1] == 0:
                Vpos.remove((el[2], el[0]))

        for insertion in ins:
            V, W = ins_repair(V, W, map, insertion, V_n, ins, Vpos)

        for deletion in d:
            V, W = del_repair(V, W, map, deletion)

        # aggiorna le label dei nodi in base a quanto contenuto nel mapping
        W_new, V_new = update_label(W, map, V)

        # riordina le liste di nodi e archi in base agli id
        V_new.sort()
        # correzione
        W_new = list(set(W_new))
        W_new.sort()

        save_g_file(V_new, W_new, join(BIG_IGS_PATH, f'{n - 1}_.g'), time() - trace_start_time, sort_labels)


def find_dependency(V, W, tipo='ig'):
    edges = W.copy()
    edges.sort(key=lambda x: x[1])
    dep = {}
    if tipo == 'net':
        j = 1
    else:
        j = 0
    for n in V:
        dep[n[j]] = []

    for e in edges:
        n1 = e[0][j]  # node's input arc
        n2 = e[1][j]  # node's output arc
        for d in dep[n1]:
            if d not in dep[n2]:
                dep[n2].append(d)
        if n1 not in dep[n2]:
            dep[n2].append(n1)

    return dep


def find_dependency_net(causal_rel, net):
    V = []
    nodes = []
    ts = net.transitions
    l = list(ts)
    transitions = sorted(l, key=lambda x: x.name)
    i = 1
    for t in transitions:
        if t.label:
            V.append((i, t.label))
            nodes.append(t.label)
            i += 1

    W = []
    for cr in causal_rel:
        id_n = nodes.index(cr[0])
        n1 = V[id_n]
        id_n = nodes.index(cr[1])
        n2 = V[id_n]
        W.append((n1, n2))

    dep = find_dependency(V, W, 'net')
    return dep


def final_restore(V, W, dep):
    new_ig = W.copy()
    for w in W:
        n1 = w[0]
        n2 = w[1]
        for d in dep[n1[0]]:
            e = (V[d], n2)
            if e in new_ig:
                new_ig.remove(e)
    return new_ig


""" le tre funzioni di seguito sono manipolazioni di tutto quello che c'è sopra.
BIG originale prende l'intero log in input, poi crea per ogni traccia un IG, fa conformance e ripara.
A noi tutti questi passaggi servono scollegati e per singola traccia."""


def get_IG(trace, causal_rels):
    A = [(idx+1, event['concept:name']) for idx, event in enumerate(trace)]
    V, W = extract_instance_graph(A, causal_rels)
    return V, W


def get_aligned_IG(trace, net, initial_marking, final_marking, causal_rels):
    Aligned, A = pick_aligned_trace(trace, net, initial_marking, final_marking)
    Align = Aligned[0]
    A1 = A[0]
    # to generate the IG on the model
    compliant = compliant_trace(Align)
    # effettiva = compliant_trace(A1)
    V, W = extract_instance_graph(compliant, causal_rels)
    return V, W


def get_repaired_IG(trace, net, initial_marking, final_marking, cr):
    Aligned, A = pick_aligned_trace(trace, net, initial_marking, final_marking)
    Align = Aligned[0]
    A1 = A[0]
    map, ins = mapping(Align, A1)

    # to generate the IG on the model
    compliant = compliant_trace(Align)
    # effettiva = compliant_trace(A1)

    d = []
    V, W = extract_instance_graph(compliant, cr)

    V_n = node_number(V)
    W_n = edge_number(W)

    for element in map:  # crea lista dei nodi da cancellare
        if element[1] == 0:
            d.append(element)

    # Vpos lista dei nodi utilizzata per repair. inizialmente si rimuovono da essa i nodi relativi a deletion.
    # In seguito viene passata in input alla funzione di insertion repair e ogni attività viene inserita all'interno di Vpos
    # nella posizione di inserimento. così facendo vpos sarà sempre aggiornata ad ogni inserimento.
    # al termine delle insertion avrò la mia lista dei nodi aggiornata.

    Vpos = []
    for node in V:
        Vpos.append(node)

    for el in map:
        if el[1] == 0:
            Vpos.remove((el[2], el[0]))

    for insertion in ins:
        V, W = ins_repair(V, W, map, insertion, V_n, ins, Vpos)

    for deletion in d:
        V, W = del_repair(V, W, map, deletion)

    # aggiorna le label dei nodi in base a quanto contenuto nel mapping
    W_new, V_new = update_label(W, map, V)

    # riordina le liste di nodi e archi in base agli id
    V_new.sort()
    # correzione
    W_new = list(set(W_new))
    W_new.sort()
    return W_new


def soundness(nodes, arcs):
    # initial node: node with no input arcs
    # final node: node with no output arcs
    inode = nodes.copy()
    for arc in arcs:
        if arc[1] in inode:
            inode.remove(arc[1])
    fnode = nodes.copy()
    for arc in arcs:
        if arc[0] in fnode:
            fnode.remove(arc[0])
    # se si verifica la condizione per cui c'è più di un initial node o più di un final node allora l'ig non è sound
    if len(fnode) > 1 or len(inode) > 1:
        return False, inode, fnode

    if len(fnode) == 0 or len(inode) == 0:
        print("NO SOUND: inode and/or fnode missing!")
        return False, inode, fnode

    return True, inode[0], fnode[0]


def apply_conformance(log, process_model, causal_rels, model_activities):
    net, init_marking, final_marking = process_model[0], process_model[1], process_model[2]

    if exists(ALIGNED_CSV_FILE_PATH):
        print('Reading the aligned igs csv...')
        aligned_igs = pd.read_csv(ALIGNED_CSV_FILE_PATH, sep=';', header=0)
        aligned_igs['edges_aligned'] = aligned_igs['edges_aligned'].apply(lambda x: literal_eval(x))
    else:
        print('Starting conformance...')
        conf_check = token_replay.apply(log, net, init_marking, final_marking)

        aligned_igs = pd.DataFrame(columns=['case_id', 'trace', 'edges_aligned'])
        for idx, trace in enumerate(log):
            case_id = trace.attributes['concept:name']
            if not conf_check[idx]['trace_is_fit']:

                trace_activities = [event['concept:name'] for event in trace]
                trace_activities_in_model = all(elem in model_activities for elem in trace_activities)

                # ignoriamo le tracce le cui attività non compaiono nel modello di processo
                if trace_activities_in_model:
                    print(f'No fit case: {case_id}')
                    V, W = get_aligned_IG(trace, net, init_marking, final_marking, causal_rels)
                    save_and_draw_ig(join(ALIGNED_IGS_PATH, f'{case_id}_aligned'), V, W)
                    aligned_igs.loc[len(aligned_igs)] = [case_id, f'{trace_activities}', W]

        aligned_igs.to_csv(ALIGNED_CSV_FILE_PATH, sep=';', index=False)
    return aligned_igs


def apply_big(log, process_model, aligned_igs, causal_rels):
    big_times_df = pd.DataFrame(columns=["trace_id", "time", 'success'])
    print(f"\n\n*** BIG ***")

    for case_id, edges_aligned in zip(aligned_igs['case_id'].tolist(), aligned_igs['edges_aligned'].tolist()):
        print(f"Processing case: {case_id}")
        net, init_marking, final_marking = process_model[0], process_model[1], process_model[2]
        # CREATE PATHS FOR IGs
        big_path_noext = join(BIG_IGS_PATH, f'{case_id}_big_no_repaired')
        big_repaired_path_noext = join(BIG_IGS_PATH, f'{case_id}_big_repaired')

        trace = next(trace for trace in log if str(trace.attributes["concept:name"]) == str(case_id))
        nodes, edges_initial_big = get_IG(trace, causal_rels)
        save_and_draw_ig(big_path_noext, nodes, edges_initial_big)

        big_start_time = time()
        try:
            edges_repaired_big = get_repaired_IG(trace, net, init_marking, final_marking, causal_rels)
            big_end_time = time()
            success, time_big = True, big_end_time-big_start_time
            save_and_draw_ig(big_repaired_path_noext, nodes, edges_repaired_big)

            is_sound, _, _ = soundness(nodes, edges_repaired_big)
            if edges_repaired_big and is_sound:
                write_or_add_text_to_file(f'{big_repaired_path_noext}.g', "SOUND")
            else:
                write_or_add_text_to_file(f'{big_repaired_path_noext}.g', "NOT SOUND")
        except Exception as e:
            print("\tSome error occurred when repairing with BIG: ", e)
            success, time_big = False, 0

        big_times_df.loc[len(big_times_df)] = [case_id, time_big, success]
        big_times_df.to_csv(BIG_TIMES_FILE_PATH, sep=';', index=False)


