__version__ = "dev1"
from maggma.stores import MongoStore
from maggma.builders import MapBuilder
from pymatgen.core.structure import Structure
from pymatgen.core.periodic_table import Element
from pymatgen.analysis.diffusion.neb.full_path_mapper import MigrationGraph
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import (
    LocalGeometryFinder,
)
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometries import *
from pymatgen.analysis.chemenv.coordination_environments.structure_environments import (
    LightStructureEnvironments,
)
from pymatgen.analysis.chemenv.connectivity import (
    structure_connectivity,
    connectivity_finder,
)
from pymatgen.analysis.chemenv.coordination_environments import (
    chemenv_strategies,
    structure_environments,
)
import networkx as nx
import numpy as np
from collections import deque


class OverlapScreeningBuilder(MapBuilder):
    """
    Look for existing periodic paths with overlapping polyhedra
    Args:
    migration_graph_store (Store): store of electrodes doc with
        migration_graph (output of migration graph builder)
    specie (str): specie of interest
    coordination ([int]): coordination number for the wi sites
    overlap ([int]): degree of overlap to look for between polyhedral wi sites
    distance_cutoff (float): distance between polyhedra, cutoff to determine
        structure connectivity
    Call overlap_find() to return (battery_id, paths)
    """

    def __init__(
        self,
        migration_graph_store: MongoStore,
        electrode_store: MongoStore,
        target_store: MongoStore,
        specie: str,
        cn: [int],
        overlap_cn: [int],
        distance_cutoff: float = 2,
        query: dict = None,
    ):
        self.migration_graph_store = migration_graph_store
        self.electrode_store = electrode_store
        self.target_store = target_store
        self.query = query
        self.specie = specie
        self.cn = cn
        self.overlap_cn = overlap_cn
        self.distance_cutoff = distance_cutoff
        
        super().__init__(source=migration_graph_store, target=target_store, query=query)
        self.connect()
        electrode_store.connect()

    def intersect_sites(self, sites1, sites2, edge_data):
        intersect = []
        for site1 in sites1:
            coords1 = site1.frac_coords + edge_data
            for site2 in sites2:
                coords2 = site2.frac_coords + edge_data
                if site1.specie == site2.specie and (
                    all(np.isclose(site1.frac_coords, coords2))
                    or all(np.isclose(coords1, site2.frac_coords))
                ):
                    intersect.append(site1)
        return intersect

    def has_cycle(self, graph, struct_graph, v):
        # BFS traversal for cycle detection
        visited = {i: False for i in graph.nodes}
        visited[v] = True
        q = deque()
        q.append((v, -1))
        periodic = {}
        migration_graph = struct_graph.graph.to_undirected()
        sites = struct_graph.structure.sites
        while q:
            (v, parent) = q.popleft()
            node1 = [i for i in range(len(sites)) if v.central_site == sites[i]][0]
            for u in graph.neighbors(v):
                node2 = [i for i in range(len(sites)) if u.central_site == sites[i]][0]
                if migration_graph.has_edge(node1, node2):
                    for i in migration_graph.get_edge_data(node1, node2).values():
                        if i["to_jimage"] != (0, 0, 0):
                            periodic[
                                tuple(sorted((u.i_central_site, v.i_central_site)))
                            ] = np.asarray(i["to_jimage"])
                if not visited[u]:
                    visited[u] = True
                    q.append((u, v))
                elif u != parent:
                    if len(periodic) > 0 and all(
                        ~np.isclose(
                            np.linalg.svd(np.row_stack(list(periodic.values())))[1], 0
                        )
                    ):
                        return True
        return False

    def overlap_find(self, mg):
        try:
            if mg is None:
                return
            struct = Structure.from_dict(mg["structure"])
            lgf = LocalGeometryFinder()
            lgf.setup_structure(struct)
            struct_graph = MigrationGraph.from_dict(mg).m_graph
            sites_from_mg = struct_graph.structure.sites
            # compute environments, and get structure connectivity
            se = lgf.compute_structure_environments(
                maximum_distance_factor=self.distance_cutoff,
                only_atoms=[self.specie],  # must identify specie interested in
                max_cn=max(self.cn),
                min_cn=min(self.cn),
                minimum_angle_factor=0.05,
            )
            strategy1 = chemenv_strategies.SimplestChemenvStrategy(
                se, distance_cutoff=1.4, angle_cutoff=0.2
            )
            strategy2 = chemenv_strategies.SimplestChemenvStrategy(
                se, distance_cutoff=1.4, angle_cutoff=0.4
            )
            strategy3 = chemenv_strategies.SimplestChemenvStrategy(
                se, distance_cutoff=1.4, angle_cutoff=0.6
            )
            strategy4 = chemenv_strategies.SimplestChemenvStrategy(
                se, distance_cutoff=2, angle_cutoff=0.2
            )
            strategy5 = chemenv_strategies.SimplestChemenvStrategy(
                se, distance_cutoff=2, angle_cutoff=0.4
            )
            strategy6 = chemenv_strategies.SimplestChemenvStrategy(
                se, distance_cutoff=2, angle_cutoff=0.6
            )
            strategies = [
                strategy1,
                strategy2,
                strategy3,
                strategy4,
                strategy5,
                strategy6,
            ]
            lse = max(
                [
                    LightStructureEnvironments.from_structure_environments(
                        strategy=strategy, structure_environments=se
                    )
                    for strategy in strategies
                ],
                key=lambda x: len(
                    [i for i in x.coordination_environments if i is not None]
                ),
            )
            connFinder = connectivity_finder.ConnectivityFinder()
            structConnectivty = connFinder.get_structure_connectivity(lse)

            # filter connectivty graph based on migration_graph
            env_graph = structConnectivty.environment_subgraph()
            env_graph = nx.Graph(env_graph)

            # nx.draw_networkx(struct_graph.graph.to_undirected())
            # print(struct_graph)
            for node in structConnectivty.environment_subgraph().nodes:
                for (n1, n2, data,) in (
                    structConnectivty.environment_subgraph()
                    .copy()
                    .edges(node, data=True)
                ):
                    node1 = [
                        i
                        for i in range(len(sites_from_mg))
                        if n1.central_site == sites_from_mg[i]
                    ][0]
                    node2 = [
                        i
                        for i in range(len(sites_from_mg))
                        if n2.central_site == sites_from_mg[i]
                    ][0]
                    if env_graph.has_edge(
                        n1, n2
                    ) and not struct_graph.graph.to_undirected().has_edge(
                        node1, node2
                    ):
                        env_graph.remove_edge(n1, n2)

            # nx.draw_networkx(env_graph)

            # filter coordination species along path
            anions = [
                Element("O"),
                Element("S"),
                Element("Se"),
                Element("F"),
                Element("Cl"),
                Element("Br"),
            ]
            paths = []
            invalid_nodes = []
            for comp in nx.connected_components(env_graph):
                connected_comp = {}
                for node in comp:
                    curr_cn = int(node.ce_symbol[-1])
                    for dist in np.linspace(1, 3, 400):
                        neigh = struct.get_neighbors(node.central_site, dist)
                        if len([i for i in neigh if i.specie in anions]) < curr_cn:
                            continue
                        if (
                            len([i for i in neigh if i.specie == Element("O")])
                            > curr_cn
                        ):
                            break
                        connected_comp[node] = [
                            i for i in neigh if i.specie in anions
                        ]
                        break
                    if node not in connected_comp:
                        invalid_nodes.append(node)
                # get the octahedrons along each connected path
                paths.append(connected_comp)

            # print([i.i_central_site for i in invalid_nodes])

            for node in invalid_nodes:
                env_graph.remove_node(node)

            # nx.draw_networkx(env_graph)

            # remove paths without tetrahedral overlap
            edge_jimages = {}
            for path in paths:
                for n1 in path:
                    for n2 in path:
                        if env_graph.has_edge(n1, n2):
                            node1 = [
                                i
                                for i in range(len(sites_from_mg))
                                if n1.central_site == sites_from_mg[i]
                            ][0]
                            node2 = [
                                i
                                for i in range(len(sites_from_mg))
                                if n2.central_site == sites_from_mg[i]
                            ][0]
                            # check for tetrahedral overlap, if not, remove edge
                            edge_data = [
                                i["to_jimage"]
                                for i in struct_graph.graph.to_undirected()
                                .get_edge_data(node1, node2)
                                .values()
                            ]
                            if all(
                                [
                                    len(
                                        self.intersect_sites(
                                            path[n1], path[n2], data
                                        )
                                    )
                                    not in self.overlap_cn
                                    for data in edge_data
                                ]
                            ):
                                env_graph.remove_edge(n1, n2)
                            else:
                                edge_jimages[(n1, n2)] = {
                                    "to_jimage": edge_data,
                                }

            nx.set_edge_attributes(env_graph, edge_jimages)
            
            for node in env_graph.copy().nodes:
                if not self.has_cycle(env_graph, struct_graph, node):
                    env_graph.remove_node(node)
            return env_graph
        except:
            return None
        
    def get_items(self) -> dict:
        """
        get info from electrode store for post-screening filter
        """
        for item in super(OverlapScreeningBuilder, self).get_items():
            eds = self.electrode_store.query_one({"battery_id" : {"$in": [item["battery_id"]]}})
            item["electrode_doc"] = eds
            yield item
        
    def unary_function(self, item: dict) -> dict:
        new_item = dict(item)
        result = self.overlap_find(item["migration_graph"])
        new_item["overlap_"+"+".join(map(str,self.cn))+"cn_"+"+".join(map(str,self.overlap_cn))+"o"] = not (result is None or len(result.nodes) == 0)
        new_item["overlap_graph_"+"+".join(map(str,self.cn))+"cn_"+"+".join(map(str,self.overlap_cn))+"o"] = result is None ? None : nx.to_dict_of_dicts(result)
        return new_item
