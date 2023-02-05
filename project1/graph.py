from scipy.special import gammaln

import graphviz
import itertools
import matplotlib.pyplot as plt
import networkx as nx
import os
import pandas as pd
import pydot
import tempfile

class Graph:
    """
    This class is used to create, modify and perform operations on graphs.
    """

    def __init__(self, infile):
        """
        This function initiates the graph object using a pandas dataframe. It 
        sets the networkx graph and dataframe as attributes, and adds nodes
        corresponding to the dataframe's column names. 
        """
        self.graph = nx.DiGraph()
        self.data  = pd.read_csv(infile)
        self.graph.add_nodes_from(list(self.data.columns))


    def compute_bayesian_score_node(self, node):
        """
        This function takes a node as argument to compute its bayesian score.
        """
        r = max(self.data[node])
        q = itertools.product(*[self.data[parent].unique() for parent in list(self.graph.pred[node])])

        score = 0

        for instance in q:
            query = ""
            for i, parent in enumerate(self.graph.pred[node]):
                query = f"{parent} == {instance[i]}" if i == 0 else f"{query} & {parent} == {instance[i]}"
            
            temp_data = self.data.query(query) if query != "" else self.data
            score += gammaln(r) - gammaln(r + len(temp_data))

            for node_instance in range(r):
                m = len(temp_data[temp_data[node] == node_instance + 1])
                score += gammaln(1 + m)

        return score


    def compute_bayesian_score(self):
        """
        This function computes the bayesian score of the whole graph using the
        helper function compute_bayesian_score_node.
        """
        return sum(self.compute_bayesian_score_node(node) for node in self.graph)


    def K2_search(self):
        """
        This function implements the K2 algorithm to modify the graph's edges
        until it is optimized.
        """

        local_scores = [self.compute_bayesian_score_node(node) for node in self.graph]
        
        for i, node in enumerate(self.graph):
            print("K2 Search: node {} / {}".format(i+1, len(self.graph)))

            while True:
                best_local_score = local_scores[i]
                best_parent      = None

                for parent in self.graph:

                    if parent == node                  or \
                        parent in self.graph.pred[node] or \
                        node in self.graph.pred[parent] or \
                        parent == self.data.columns[-1]:
                        continue
                    
                    self.graph.add_edge(parent, node)
                    local_score = self.compute_bayesian_score_node(node)

                    if nx.is_directed_acyclic_graph(self.graph) and \
                        local_score > best_local_score:
                        best_local_score, best_parent = local_score, parent

                    self.graph.remove_edge(parent, node)

                if best_parent:
                    local_scores[i] = best_local_score
                    self.graph.add_edge(best_parent, node)

                else:
                    break
            
        final_score = sum(local_scores)
        print("\n" + "Score: ", final_score)
        print("----------------------")

    
    def export_bayesian_score(self, outfile):
        """
        This function is used to output the bayesian score of the graph to a
        .score file using the outfile string as output path.
        """
        with open(outfile + ".score", 'w') as f:
            f.write(str(self.compute_bayesian_score()))
        

    def write_gph(self, filename):
        """
        This function writes the graph to a .gph file using:
        - filename: file name for the output file
        """
        with open(filename + ".gph", 'w') as f:
            for edge in self.graph.edges():
                f.write("{}, {}\n".format(edge[0], edge[1]))


    def draw_gph(self, filename):
        """
        This function takes the current graph and exports it as a .png file using
        the filename variable as the destination path.
        """
        with tempfile.NamedTemporaryFile() as f:
            nx.drawing.nx_pydot.write_dot(self.graph, f.name)
            img_file = graphviz.render('dot', 'png', f.name)
            os.rename(img_file, filename + ".png")

        