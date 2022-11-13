import networkx as nx


def has_cycles(edges):
    print("\nDETECTING cycles in %s" % (edges))
    G = nx.DiGraph(edges)

    cycles = False
    for cycle in nx.simple_cycles(G):
        print("Cycle found:"+str(cycle))
        cycles = True

    if cycles is False:
        print("No cycles found!")
    return cycles


G1 = [('A', 'B'), ('A', 'D'), ('B', 'C'), ('C', 'D'), ('D', 'E'), ('E', 'B')]
G2 = [('A', 'B'), ('A', 'D'), ('B', 'C'), ('D', 'C'), ('D', 'E'), ('E', 'B')]
has_cycles(G1)
has_cycles(G2)
