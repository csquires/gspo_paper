import itertools as itr
import causaldag as cd
from collections import defaultdict
from copy import deepcopy


class IntransitiveParentError(Exception):
    def __init__(self, i, j):
        message = f"The relation {i}<{j} cannot be added since there is a k s.t. k<{i} but k is not <{j}"
        super().__init__(message)

class IntransitiveChildrenError(Exception):
    def __init__(self, i, j):
        message = f"The relation {i}<{j} cannot be added since there is a k s.t. {j}<k but {i} is not <k"
        super().__init__(message)

#def visit_forward(dag, node, stack, visited, ancestor_dict):
#    #Update node's ancestors
#    ancestor_dict[node].update(stack)
#    if(node not in visited):
#        visited.add(node)
#        stack.append(node)
#        for child in dag._children[node]:
#            visit_forward(dag, child, stack, visited, ancestor_dict)
#        stack.pop(node)
#
#def visit_backward(dag, node, stack, visited, descendant_dict):
#    #Update node's ancestors
#    descendant_dict[node].update(stack)
#    if(node not in visited):
#        visited.add(node)
#        stack.append(node)
#        for parent in dag._parnts[node]:
#            visit_backward(dag, parent, stack, visited, descendant_dict)
#        stack.pop(node)

#def get_updated_ancestry_relations(dag):
#    """updates ancestors and descendants relations of all nodes in poset based on dag"""
#    sorted_nodes = dag.topological_sort()
#    #Run DFS forward
#    ancestor_dict = defaultdict(set)
#    visited = set()
#    stack = []
#    for node in sorted_nodes:
#        visit_forward(dag, node, stack, visited, ancestor_dict)
#    #Run DFS backward 
#    descendant_dict = defaultdict(set)
#    visited = set()
#    stack = []
#    for node in reversed(sorted_nodes):
#        visit_backward(dag, node, stack, visited, descendant_dict)
#    return ancestor_dict,descendant_dict

class Poset:
    def __init__(self, nodes):
        """
        Invariant: the underlying DAG should remain a Hasse diagram,
        i.e. i->j implies there is no path i->k->...->j
        """
        self.underlying_dag = cd.DAG(nodes=nodes)
        self._ancestors = defaultdict(set)
        self._descendants = defaultdict(set)
        self._num_relations = 0

    def copy(self):
        p = Poset(self.underlying_dag.nodes)
        p.underlying_dag = self.underlying_dag.copy()
        p._ancestors = deepcopy(self._ancestors)
        p._descendants = deepcopy(self._descendants)
        p._num_relations = self._num_relations
        return p

    @classmethod
    def from_dag(cls, dag):
        p = Poset(dag.nodes)
        d = dag.copy()
        while d.nodes:
            for i in d.sources():
                for j in d.downstream(i):
                    p.add_covering_relation(i, j)
                d.remove_node(i, ignore_error = True)
        full_dag = p.get_dag_transitive_closure()
        p.underlying_dag = full_dag
        return p

    def __str__(self):
        return str(self.underlying_dag._arcs)

    def __hash__(self):
        return hash((frozenset(self.underlying_dag._arcs), frozenset(self.underlying_dag._nodes)))

    def less_than(self, i, j):
        """is i < j?"""
        return i in self._ancestors[j]

    def greater_than(self, i, j):
        """is i > j?"""
        return i in self._descendants[j]

    def incomparable(self, i, j):
        return (not self.less_than(i, j)) and (not self.greater_than(i, j))

    def get_smaller(self, i):
        return self._ancestors[i].copy()

    @property
    def num_relations(self):
        return self._num_relations

    @property
    def size(self):
        return len(self.underlying_dag.nodes)

    # def add_covering_relations(self, S):
    #     for e in S:
    #         self.underlying_dag.add_arc(e[0], e[1])

    def _add_covering_relation(self, i, j):
        self.underlying_dag.add_arc(i, j)
        self._descendants[i].add(j)
        self._ancestors[j].add(i)
        self._num_relations += 1


    #def add_legitimate_relation(self, i, j, mag):
    #    error = True
    #    while(error):
    #        try:
    #            self.underlying_dag.add_arc(i,j)
    #            error=False
    #        except cd.CycleError as e:
    #            cycle = e.cycle
    #            for ind in range(len(cycle) -1): #the cycle is in i to j
    #                x = cycle[ind]
    #                y = cycle[ind+1]
    #                #Check if not expressed:
    #                if(not x==i and not y==j and not x in mag._parents[y]):
    #                    self._flip_relation(x, y, mag)
    #    self._descendants[i].add(j)
    #    self._ancestors[j].add(i)
    #    self._num_relations += 1

    def _add_covered_relation(self, i, j):
        """Method assumes that descendants(i).intersection(ancestors(j)) = phi, and i->j"""
        self.underlying_dag.remove_arc(i,j)
        self._descendants[i].remove(j) 
        self._ancestors[j].remove(i)
        #Update underlying dag
        self._num_relations -= 1

    def _flip_relation(self, i, j, mag):
        """Method assumes that parents(i) + i = parents(j)"""
        self.underlying_dag.remove_arc(i,j)
        self._descendants[i].remove(j) 
        self._ancestors[j].remove(i)

        error = True
        while(error):
            try:
                self.underlying_dag.add_arc(j,i)
                error=False
            except cd.CycleError as e:
                cycle = e.cycle
                for ind in range(len(cycle) -1): #the cycle is in j to i
                    x = cycle[ind]
                    y = cycle[ind+1]
                    #Check if not expressed:
                    if(not x==j and not y==i and not x in mag._parents[y]):
                        self._flip_relation(x, y, mag)

        self._ancestors[i].add(j)
        self._descendants[j].add(i) 


    def add_covering_relation(self, i, j):
        """add i<j. Only allowed if it adding this relation does not imply any other relations by transitivity."""
        if not self.underlying_dag._parents[i] <= self._ancestors[j]:
            raise IntransitiveParentError(i, j)
        if not self.underlying_dag._children[j] <= self._descendants[i]:
            raise IntransitiveChildrenError(i, j)
        self._add_covering_relation(i, j)

    def is_total_order(self):
        return all(
            [not self.incomparable(e[0], e[1]) for e in set(itr.combinations(self.underlying_dag.nodes, 2))])

    def get_incomparable_pairs(self):
        combinations = {(i, j) for i, j in itr.combinations(self.underlying_dag.nodes, r=2) if self.incomparable(i, j)}
        return combinations | set(map(reversed, combinations))

    def get_ordered_pairs(self):
        ordered_pairs = {(i, j) for i, j in itr.permutations(self.underlying_dag.nodes, r=2) if self.less_than(i, j)}
        return ordered_pairs

    # def get_covering_relations(self):
    #     covering_relations = []
    #     for e in self.underlying_dag.arcs:
    #         for i in self.underlying_dag.nodes:
    #             if i != e[0] and i != e[1]:
    #                 if not (self.less_than(e[0], i) and self.less_than(i, e[1])):
    #                     if not e in covering_relations:
    #                         covering_relations.append(e)
    #     return covering_relations

    def get_dag_transitive_closure(self):
        node_set = self.underlying_dag.nodes
        to_return = cd.DAG(nodes=node_set)
        for e in itr.combinations(node_set, 2):
            if self.less_than(e[0], e[1]):
                to_return.add_arc(e[0], e[1])
            elif self.less_than(e[1], e[0]):
                to_return.add_arc(e[1], e[0])
        return to_return

    def get_covering_posets(self):
        covering_posets = []
        for i, j in self.get_incomparable_pairs():
            parents_okay = self.underlying_dag._parents[i] <= self._ancestors[j]
            children_okay = self.underlying_dag._children[j] <= self._descendants[i]
            if parents_okay and children_okay:
                p = self.copy()
                p._add_covering_relation(i, j)
                covering_posets.append(p)
        return covering_posets

    def get_covered_posets(self):
        covered_posets = []
        for i,j in self.get_ordered_pairs():
            if(len(self._descendants[i].intersection(self._ancestors[j]))==0):
                p = self.copy()
                p._add_covered_relation(i,j)
                covered_posets.append(p)
        return covered_posets

    def __eq__(self, other):
        if self.underlying_dag.nodes != other.underlying_dag.nodes:
            return False
        for e in itr.combinations(self.underlying_dag.nodes, 2):
            u = e[0]
            v = e[1]
            if self.less_than(u, v) and not other.less_than(u, v):
                return False
            if other.less_than(u, v) and not self.less_than(u, v):
                return False
            if self.greater_than(u, v) and not other.greater_than(u, v):
                return False
            if other.greater_than(u, v) and not self.greater_than(u, v):
                return False
        return True

if __name__ == '__main__':
    dag = cd.DAG(arcs={(0, 1), (1, 3), (3, 4), (2, 3), (0, 3), (0, 4)})
    p = Poset.from_dag(dag)
    # VERBOSE = False
    # empty_poset = Poset(4)
    #
    # visited_posets = {frozenset(empty_poset.underlying_dag._arcs)}
    # queue = [empty_poset]
    # while queue:
    #     current_poset = queue.pop(0)
    #     covering_posets = current_poset.get_covering_posets()
    #     for poset in covering_posets:
    #         arcs = frozenset(poset.underlying_dag._arcs)
    #         # if arcs == {(1, 0), (0, 2), (1, 2)}:
    #         #     print(current_poset.underlying_dag.arcs)
    #         if arcs not in visited_posets:
    #             queue.append(poset)
    #             visited_posets.add(arcs)
    #
    # v = list(sorted(visited_posets, key=lambda p: len(p)))
    # print(len(visited_posets))

    # p = Poset(cd.DAG(nodes=set(range(3)), arcs={(0, 2)}))
    # print([t.underlying_dag.arcs for t in p.get_covering_posets()])

    # p = Poset(cd.DAG(nodes=set(range(3)), arcs={(1, 0), (0, 2)}))
    # print([t.underlying_dag.arcs for t in p.get_covering_posets(verbose=True)])

