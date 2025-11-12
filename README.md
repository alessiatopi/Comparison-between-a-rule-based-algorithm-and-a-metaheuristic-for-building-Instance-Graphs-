# Comparison-between-a-rule-based-algorithm-and-a-metaheuristic-for-building-Instance-Graphs-from-event-logs
My master’s thesis focused on the comparison between a rule-based algorithm and a metaheuristic for the generation of Instance Graphs from event logs.

I’ve included a few Python scripts to showcase the core logic of the project, which was part of a larger university research effort involving multiple contributors.

The **metaheuristic, inspired by Simulated Annealing, iteratively explores alternative graph structures** through:

-*Initialization* – ensuring soundness of the starting graph;

-*Local Search* – random edge removal/addition to explore neighboring solutions;

-*Objective Function* – balancing generalization, structural changes, and alignment with the process model;

-*Probabilistic Acceptance* – allowing worse solutions early to escape local minima.

My contribution focused on **parameter tuning, correcting code inconsistencies and the development of the tabella.py, res_combination_seeds.py, res_sim_all.py, res_visualize_trace_metrics.py scripts** for automated evaluation and comparison of both algorithms. 

The goal was to identify when the metaheuristic outperforms BIG, given its higher computational cost.
