# Standard Algorithm Implementation
# Sampling-based Algorithms PRM

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from scipy import spatial
import time

# Class for PRM


class PRM:
    # Constructor
    def __init__(self, map_array):
        self.map_array = map_array            # map array, 1->free, 0->obstacle
        self.size_row = map_array.shape[0]    # map size
        self.size_col = map_array.shape[1]    # map size

        self.samples = []                     # list of sampled points
        self.graph = nx.Graph()               # constructed graph
        self.path = []                        # list of nodes of the found path

        self.best_samples = []
        self.best_graph = nx.Graph()
        self.best_path = []
        self.best_length = np.inf
        self.best_time = np.inf

        self.paths_length = []
        self.build_time = []

    def check_collision(self, p1, p2):
        '''Check if the path between two points collide with obstacles
        arguments:
            p1 - point 1, [row, col]
            p2 - point 2, [row, col]

        return:
            True if there are obstacles between two points
        '''

        delta = 1  # spacing between each sample point
        x1, y1 = p1
        x2, y2 = p2
        theta = np.arctan2(y2-y1, x2-x1)  # slope of the line
        sx = x1
        sy = y1
        for i in range(1, round(self.dis(p1, p2))+1):
            sx = round(x1+i*delta*np.cos(theta))
            sy = round(y1+i*delta*np.sin(theta))
            if not self.map_array[sx][sy]:
                return True
        return False

    def dis(self, point1, point2):
        '''Calculate the euclidean distance between two points
        arguments:
            p1 - point 1, [row, col]
            p2 - point 2, [row, col]

        return:
            euclidean distance between two points
        '''

        return np.linalg.norm(np.array(point1)-np.array(point2))

    def uniform_sample(self, n_pts):
        '''Use uniform sampling and store valid points
        arguments:
            n_pts - number of points try to sample, 
                    not the number of final sampled points

        check collision and append valide points to self.samples
        as [(row1, col1), (row2, col2), (row3, col3) ...]
        '''
        # Initialize graph
        self.graph.clear()

        row = np.linspace(0, self.size_row-1,
                          num=int(np.sqrt(n_pts)), dtype='int32')
        col = np.linspace(0, self.size_col-1,
                          num=int(np.sqrt(n_pts)), dtype='int32')
        for r in row:
            for c in col:
                p = (r, c)
                if self.map_array[p[0], p[1]]:
                    self.samples.append(p)

    def random_sample(self, n_pts):
        '''Use random sampling and store valid points
        arguments:
            n_pts - number of points try to sample, 
                    not the number of final sampled points

        check collision and append valide points to self.samples
        as [(row1, col1), (row2, col2), (row3, col3) ...]
        '''
        # Initialize graph
        self.graph.clear()
        i = 1

        while i < n_pts:
            p = np.random.randint([self.size_row-1, self.size_col-1])
            if self.map_array[p[0], p[1]]:
                self.samples.append(tuple(p))
                i += 1

    def gaussian_sample(self, n_pts, std_dev=10, p_random=0.2):
        '''Use gaussian sampling and store valid points
        arguments:
            n_pts - number of points try to sample, 
                    not the number of final sampled points

        check collision and append valide points to self.samples
        as [(row1, col1), (row2, col2), (row3, col3) ...]
        '''
        # Initialize graph
        self.graph.clear()
        i = 1

        while i < n_pts:
            # with probability p_random, take a uniform random free sample
            if np.random.rand() < p_random:
                p = np.random.randint([self.size_row-1, self.size_col-1])
                if self.map_array[p[0], p[1]]:
                    self.samples.append(tuple(p))
                    i += 1
                continue

            # gaussian-biased sampling towards obstacle boundaries
            p1 = np.random.uniform(
                [self.size_row-1, self.size_col-1]).astype('int32')
            p2 = np.random.normal(p1, std_dev).astype('int32')
            if (0 <= p2[0] < self.size_row) & (0 <= p2[1] < self.size_col):
                if (self.map_array[p1[0], p1[1]] ^ self.map_array[p2[0], p2[1]]):
                    if self.map_array[p1[0], p1[1]]:
                        self.samples.append(tuple(p1))
                    else:
                        self.samples.append(tuple(p2))
                    i += 1

    def bridge_sample(self, n_pts, std_dev=20, p_random=0.2):
        '''Use bridge sampling and store valid points
        arguments:
            n_pts - number of points try to sample, 
                    not the number of final sampled points

        check collision and append valide points to self.samples
        as [(row1, col1), (row2, col2), (row3, col3) ...]
        '''
        # Initialize graph
        self.graph.clear()
        i = 1

        while i < n_pts:
            # with probability p_random, take a uniform random free sample
            if np.random.rand() < p_random:
                p = np.random.randint([self.size_row-1, self.size_col-1])
                if self.map_array[p[0], p[1]]:
                    self.samples.append(tuple(p))
                    i += 1
                continue

            # bridge sampling biased towards narrow passages / obstacle boundaries
            p1 = np.random.uniform(
                [self.size_row-1, self.size_col-1]).astype('int32')
            if not self.map_array[p1[0], p1[1]]:
                p2 = np.random.normal(p1, std_dev).astype('int32')
                if (0 <= p2[0] < self.size_row) & (0 <= p2[1] < self.size_col) and not self.map_array[p2[0], p2[1]]:
                    mid_pt = tuple(np.mean([p1, p2], axis=0, dtype='int32'))
                    if self.map_array[mid_pt[0], mid_pt[1]]:
                        self.samples.append(mid_pt)
                        i += 1

    def draw_map(self, samples, path, graph, save_path=None):
        '''Visualization of the result
        arguments:
            save_path - optional path to save the drawn figure as PNG
        '''
        # Create empty map
        fig, ax = plt.subplots()
        img = 255 * np.dstack((self.map_array, self.map_array, self.map_array))
        ax.imshow(img)

        # Draw graph
        # get node position (swap coordinates)
        node_pos = np.array(samples)[:, [1, 0]]
        pos = dict(zip(range(len(samples)), node_pos))
        pos['start'] = (samples[-2][1], samples[-2][0])
        pos['goal'] = (samples[-1][1], samples[-1][0])

        # draw constructed graph
        nx.draw(graph, pos, node_size=3,
                node_color='y', edge_color='y', ax=ax)

        # If found a path
        if path:
            # add temporary start and goal edge to the path
            final_path_edge = list(zip(path[:-1], path[1:]))
            nx.draw_networkx_nodes(
                graph, pos=pos, nodelist=path, node_size=8, node_color='b')
            nx.draw_networkx_edges(
                graph, pos=pos, edgelist=final_path_edge, width=2, edge_color='b')

        # draw start and goal
        nx.draw_networkx_nodes(graph, pos=pos, nodelist=[
                               'start'], node_size=12,  node_color='g')
        nx.draw_networkx_nodes(graph, pos=pos, nodelist=[
                               'goal'], node_size=12,  node_color='r')

        # show image and optionally save to file
        plt.axis('on')
        ax.tick_params(left=True, bottom=True,
                       labelleft=True, labelbottom=True)
        if save_path:
            try:
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
            except Exception:
                pass
        # plt.show()
        plt.close(fig)

    def sample(self, n_pts=1000, k=8, std_dev=20, p_random=0.2, sampling_method="uniform"):
        '''Construct a graph for PRM
        arguments:
            n_pts - number of points try to sample, 
                    not the number of final sampled points
            sampling_method - name of the chosen sampling method

        Sample points, connect, and add nodes and edges to self.graph
        '''
        # Initialize before sampling
        self.samples = []
        self.graph.clear()
        self.path = []

        # Start timer for the graph construction process
        t0 = time.perf_counter()

        # Sample methods
        if sampling_method == "uniform":
            self.uniform_sample(n_pts)
        elif sampling_method == "random":
            self.random_sample(n_pts)
        elif sampling_method == "gaussian":
            self.gaussian_sample(n_pts, std_dev, p_random)
        elif sampling_method == "bridge":
            self.bridge_sample(n_pts, std_dev, p_random)

        # Find the pairs of points that need to be connected
        # and compute their distance/weight.
        # Store them as
        # pairs = [(p_id0, p_id1, weight_01), (p_id0, p_id2, weight_02),
        #          (p_id1, p_id2, weight_12) ...]
        pairs = []
        self.kdtree = spatial.KDTree(self.samples)
        _, pairs_idx = self.kdtree.query(
            self.samples, k)
        for i in range(len(pairs_idx)):
            for j in range(1, len(pairs_idx[0])):
                p1 = self.samples[i]
                p2 = self.samples[pairs_idx[i][j]]
                if not self.check_collision(p1, p2):
                    pairs.append(
                        (pairs_idx[i][0], pairs_idx[i][j], self.dis(p1, p2)))

        # Use sampled points and pairs of points to build a graph.
        # To add nodes to the graph, use
        # self.graph.add_nodes_from([p_id0, p_id1, p_id2 ...])
        # To add weighted edges to the graph, use
        # self.graph.add_weighted_edges_from([(p_id0, p_id1, weight_01),
        #                                     (p_id0, p_id2, weight_02),
        #                                     (p_id1, p_id2, weight_12) ...])
        # 'p_id' here is an integer, representing the order of
        # current point in self.samples
        # For example, for self.samples = [(1, 2), (3, 4), (5, 6)],
        # p_id for (1, 2) is 0 and p_id for (3, 4) is 1.
        self.graph.add_nodes_from(range(1, len(self.samples)+1))
        self.graph.add_weighted_edges_from(pairs)

        # Print constructed graph information
        n_nodes = self.graph.number_of_nodes()
        n_edges = self.graph.number_of_edges()
        print("The constructed graph has %d nodes and %d edges" %
              (n_nodes, n_edges))
        # Print time it took to build the graph
        elapsed = time.perf_counter() - t0
        self.build_time.append(elapsed)
        print("Graph build time: %.4f seconds" % elapsed)

    def search(self, start, goal, k=5, show_drawing=True):
        '''Search for a path in graph given start and goal location
        arguments:
            start - start point coordinate [row, col]
            goal - goal point coordinate [row, col]

        Temporary add start and goal node, edges of them and their nearest neighbors
        to graph for self.graph to search for a path.
        '''
        # Clear previous path
        self.path = []

        # Temporarily add start and goal to the graph
        self.samples.append(start)
        self.samples.append(goal)
        # start and goal id will be 'start' and 'goal' instead of some integer
        self.graph.add_nodes_from(['start', 'goal'])

        # Find the pairs of points that need to be connected
        # and compute their distance/weight.
        # You could store them as
        # start_pairs = [(start_id, p_id0, weight_s0), (start_id, p_id1, weight_s1),
        #                (start_id, p_id2, weight_s2) ...]
        start_pairs = []
        goal_pairs = []
        _, start_pairs_idx = self.kdtree.query(start, k)
        _, goal_pairs_idx = self.kdtree.query(goal, k)
        for s_idx in start_pairs_idx:
            p = self.samples[s_idx]
            if not self.check_collision(start, p):
                start_pairs.append(("start", s_idx, self.dis(start, p)))
        for g_idx in goal_pairs_idx:
            p = self.samples[g_idx]
            if not self.check_collision(goal, p):
                goal_pairs.append(("goal", g_idx, self.dis(goal, p)))
        # Add the edge to graph
        self.graph.add_weighted_edges_from(start_pairs)
        self.graph.add_weighted_edges_from(goal_pairs)

        # Seach using Dijkstra
        try:
            self.path = nx.algorithms.shortest_paths.weighted.dijkstra_path(
                self.graph, 'start', 'goal')
            path_length = nx.algorithms.shortest_paths.weighted.dijkstra_path_length(
                self.graph, 'start', 'goal')
            print("The path length is %.2f" % path_length)
            self.paths_length.append(path_length)
        except nx.exception.NetworkXNoPath:
            print("No path found")
            path_length = np.inf
            self.paths_length.append(np.float64(0))

        if self.best_length > path_length:
            self.best_length = path_length
            self.best_time = self.build_time[-1]
            self.best_path = self.path.copy()
            self.best_graph = self.graph.copy()
            self.best_samples = self.samples.copy()

        # Draw result
        if show_drawing:
            self.draw_map(self.samples, self.path, self.graph)
