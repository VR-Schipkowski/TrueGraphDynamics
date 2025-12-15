from datetime import datetime
from enum import unique
import pandas as pd
import regex as re
from pyvis.network import Network
import networkx as nx
import tqdm
import random

class TimeIntervall:
    def __init__(self,time_data):
        
        self.start_time = datetime(9999, 12, 31, 23, 59, 59) 
        self.end_time = datetime(1, 1, 1, 0, 0, 0)
        try:
            for data in time_data:
                if data >= self.end_time:
                    self.end_time =data
                if data<= self.start_time:
                    self.start_time=data
            print(f"the time intervall of the data goes\nfrom: {self.start_time} \nto: {self.end_time}")
        except:
            print("Error there was an issue setting the time interall!!!")

class Follower_Edge:
    #follows.tsv
    def __init__(self,edge_entry):
        self.id = edge_entry[0]
        self.time_scraped

class Truth:
    def __init__(self,truth_entry):
        try:
            self.timestamp= truth_entry[1]
            self.text = truth_entry[9]
            self.like_count= truth_entry[6]
            self.retruth_count= truth_entry[7]
            self.reply_count= truth_entry[8]
        except:
            print("there was an error reading the following truth id:")
            print(truth_entry)
class Node:
    #users.tsv
    def __init__(self ,entry):
        entry= entry.split("\t")
        
        self.id=entry[0]
        self.timestamp=entry[1]
        self.time_scraped=entry[2]
        self.username=entry[3]
        self.follower_count=entry[4]
        self.following_count=entry[5]
        self.follower_edges=[]
        self.following_edges=[]
        self.truths={}

    def node_info(self):
        print(f"Information about node {self.id}:")
        print(f"Username: {self.username}")
        print(f"follower count: {self.follower_count}")
        print(f"following count: {self.following_count}")
        #print(f"following users: {self.following_edges}")

    def add_following_edge(self,edge_entry):
        #still need to decide if this is necessary -> means that we need to save data twice
        edge= Follower_Edge(edge_entry)
        self.following_edges[edge.id]=edge

    def add_follower_edge(self,edge_entry):
        edge= Follower_Edge(edge_entry)
        self.follower_edges[edge.id]=edge
        
    def edges_created_before(self,timestamps):
        print(timestamps)#TODO:this function is not done 

    def add_truth(self,truth_entry):
            self.truths[truth_entry[0]]=Truth(truth_entry)


class TemporalGraph:
    def __init__(self,nodes_file="../Data/OriginalData/users.tsv",follower_file="../Data/OriginalData/follows.tsv",truths_file="../Data/OriginalData/truths.tsv",time_intervall_file="../Data/OriginalData/truths.tsv"):
        """
        Docstring for __init__
        
        :param self: Description
        :param nodes_file: accepts either tsv or csv file of form #TODO:
        :param follower_file: accepts either tsv or csv file of form #TODO:
        :param truth_file: accepts either tsv or csv file of form #TODO:
        :param time_intervall_file: accepts either tsv or csv file of form #TODO:
        

        """
        self.nodes_dict={}
        self.create_nodes_from_file(nodes_file)
        self.create_follower_edges_from_file(follower_file)
        self.assign_truths_to_nodes(truths_file)
        self.time_intervall=self.create_timeintervall(time_intervall_file)
        self.assign_follower_edges()

        self.active_nodes=[]

        
    def add_node(self,node_entry):
        node = Node(node_entry)
        if node.id in self.nodes_dict:
            print("node already exists:"+node.id)
        else:
            self.nodes_dict[node.id]=node
    
    def create_nodes_from_file(self,filename):
        print("Creating nodes from file ...")
        if filename.endswith(".tsv"):
            with open(filename, "r") as file:
                lines = file.readlines()
                lines=lines[1:]  # returns a list of all lines
                for line in lines:
                    self.add_node(line)
        elif filename.endswith(".csv"):
            print("the csv file format is not supportet right now")
        else:
            print(f"the {filename.split(".")[-1]} file format is not supported ")
        print("done.")

    def assign_follower_edges(self):
        print("assigning following edges...")
        node_ids = list(self.nodes_dict.keys())
        counter=0
        for node_id in node_ids:
            for following_node_id in self.nodes_dict[node_id].following_edges:
                #print(following_node_id)
                #if str(following_node_id) in node_ids:
                try:
                    self.nodes_dict[following_node_id].follower_edges.append(node_id)
                    #print(f"This is a match: {following_node_id}")
                except:
                    counter+=1
        print(f"{counter} number of pars could not be connected")

    def create_follower_edges_from_file(self,filename):
        print("creating follower edges from file ...")
        if filename.endswith(".tsv"):
            with open(filename,"r") as file:
                lines = file.readlines()
                lines= lines[1:]
                unique_followers=[]
                for line in lines:
                    line=line.split("\t")
                    self.nodes_dict[line[2]].following_edges.append(re.sub(r"\D+", "",line[3] ))#TODO: this is not finished
                    
                    unique_followers.append(line[2])
        elif filename.endswith(".csv"):
            print("the csv file format is not supportet right now")
        else:
            print(f"the {filename.split(".")[-1]} file format is not supported ")
        print("number of unique followers:"+str(len(set(unique_followers))))
        print("done.")

    def nodes_created_before(self,timestamps):
        for node_id in self.nodes_dict.items():
            for timestamp in timestamps:
                if self.nodes_dict[node_id].timestamp <= timestamp :
                    self.nodes_at_timestamp = node_id
                    break 
    
    def assign_truths_to_nodes(self,filename):
        print("assigning truths to nodes ...")
        if filename.endswith(".tsv"):
            with open(filename,"r", encoding="utf-8") as file:
                    lines=file.readlines()
                    lines=lines[1:]
                    messages_droped=0
                    for line in lines:
                            
                            try:
                                line=line.split("\t")
                                self.nodes_dict[line[5]].add_truth(line)
                            except:
                                #print("there was an error:")
                                #print(line[5])
                                if line[5] not in self.nodes_dict.items():
                                    messages_droped+=1
                    print("mesages dropped due to unforseen circumstancers: "+ str(messages_droped))#TODO:better explanation         
        elif filename.endswith(".csv"):
            print("the csv file format is not supportet right now")
        else:
            print(f"the {filename.split(".")[-1]} file format is not supported ")
        print("done.")

    def create_timeintervall(self,filename):
        print("creating time intervall ...")
        if filename.endswith(".tsv"):
            with open(filename,"r", encoding="utf-8") as file:
                lines=file.readlines()
                lines=lines[1:]
                lines=[line.split("\t")[1] for line in lines]
                lines = pd.to_datetime(lines, format='mixed', dayfirst=False,errors='coerce')
                TimeIntervall(lines.to_pydatetime())
        elif filename.endswith(".csv"):
            print("the csv file format is not supportet right now")
        else:
            print(f"the {filename.split(".")[-1]} file format is not supported ")
        print("done.")

    def get_nodes_with_dist_n(self,n,node_id):
        """
        Docstring for get_nodes_with_dist_n
        :param self: Deinition
        :param n: distance between node[node_id] and collected nodes
        :param node_id: id of node we want to collect the distant neighbours from 
        """
        node_ids=[node_id]
        final_node_ids=node_ids
        for _ in range(0,n):
            new_node_ids=[]
            for node in node_ids:
                new_node_ids=list(set(new_node_ids)| set(self.nodes_dict[node].following_edges))
            node_ids=[x for x in new_node_ids if x not in final_node_ids]
            print("final_node_ids:"+str(final_node_ids))
            final_node_ids.extend(node_ids)

        return final_node_ids
    
    def check_if_followings_exist(self,change):
        node_list=list(self.nodes_dict.keys())
        change=False
        for node_id in node_list:
            for element in list(self.nodes_dict[node_id].following_edges):
                if element not in node_list:
                    self.nodes_dict[node_id].following_edges.remove(element)
                    change =True
        return change 
            

    def clean_nodes(self,no_truths=True ,no_followings=True, repeat=True ):
        print("cleaning nodes ...")
        if repeat == True:
            iteration_counter = 1
            while repeat == True: 
                print(f"{iteration_counter}. Iteration of cleaning ...")
                iteration_counter+=1
                for node_id in list(self.nodes_dict.keys()):
                    if no_truths and len(self.nodes_dict[node_id].truths)==0 :#and len(self.nodes_dict[node_id].follower_edges)==0
                        self.nodes_dict.pop(node_id)
                    elif no_followings and len(self.nodes_dict[node_id].following_edges)==0 and len(self.nodes_dict[node_id].follower_edges)==0:
                        self.nodes_dict.pop(node_id)
                repeat = self.check_if_followings_exist(repeat)
        elif repeat == False:
            for node_id in list(self.nodes_dict.keys()):
                if no_truths and len(self.nodes_dict[node_id].truths)==0 :#and len(self.nodes_dict[node_id].follower_edges)==0
                    self.nodes_dict.pop(node_id)
                elif no_followings and len(self.nodes_dict[node_id].following_edges)==0 and len(self.nodes_dict[node_id].follower_edges)==0:
                    self.nodes_dict.pop(node_id)
        #TODO: take into account if nodes are followers to diferent nodes 
        print("nodes cleaned ...")

    def create_pyvis_representation(self, sample_size=3000, visualize_scc=False):
    
        print("Creating directed graph from nodes_dict...")
        Graph = nx.DiGraph()
        
        for node_id in tqdm.tqdm(list(self.nodes_dict.keys())):
            for following_node_id in self.nodes_dict[node_id].following_edges:
                Graph.add_edge(node_id, following_node_id)
        
        print("Computing connectivity...")

        # Weak connectivity (ignores edge directions)
        wcc = list(nx.weakly_connected_components(Graph))
        num_wcc = len(wcc)
        largest_wcc = max(wcc, key=len)
        
        print("Weak connectivity:")
        print(f"Number of weakly connected components: {num_wcc}")
        print(f"Largest component size: {len(largest_wcc)}")

        # Strong connectivity (respects edge directions)
        scc = list(nx.strongly_connected_components(Graph))
        num_scc = len(scc)
        largest_scc = max(scc, key=len)
        
        print("Strong connectivity:")
        print(f"Number of strongly connected components: {num_scc}")
        print(f"Largest SCC size: {len(largest_scc)}")

        # If requested, visualize a manageable subgraph
        if sample_size and sample_size < len(Graph):
            print(f"Sampling {sample_size} nodes for visualization...")
            sampled_nodes = random.sample(list(largest_wcc), min(sample_size, len(largest_wcc)))
            subgraph = Graph.subgraph(sampled_nodes).copy()
        else:
            subgraph = Graph  # only safe if small
        print("removing isolated vertices from subgraph...")
        isolated_nodes = list(nx.isolates(subgraph))
        subgraph.remove_nodes_from(isolated_nodes)
        print("Creating PyVis network...")
        net = Network(directed=True, notebook=False)
        net.from_nx(subgraph)

        # Performance optimization: disable physics
        net.toggle_physics(True)

        # Optional: visualize SCC condensation graph
        if visualize_scc:
            print("Creating condensation graph for SCC visualization...")
            C = nx.condensation(Graph, scc)
            net = Network(directed=True, notebook=False)
            net.from_nx(C)
            net.toggle_physics(True)

        print("Writing HTML...")
        net.write_html("directed_graph.html")
        print("Visualization created (HTML saved).")
        
        return Graph, largest_wcc, largest_scc


#graph.create_nodes_from_file("truth_social/users.tsv")
#graph.assign_truths_to_nodes()