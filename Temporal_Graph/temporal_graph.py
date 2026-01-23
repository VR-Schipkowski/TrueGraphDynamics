from datetime import datetime, timedelta
from math import e
from dateutil.relativedelta import relativedelta
from enum import unique
import pandas as pd
import regex as re
from pyvis.network import Network
import networkx as nx
import tqdm

class TimeIntervall:
    def __init__(self,time_data,spltit_by="week"):
        
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
        self.generate_time_intervals(spltit_by)    

    def generate_time_intervals(self,split_by):
        if self.start_time >= self.end_time:
            raise ValueError("start must be before end")
        
        step_map = {
            "day": timedelta(days=1),
            "week": timedelta(weeks=1),
            "month": relativedelta(months=1),
            "year": relativedelta(years=1),
        }

        if split_by not in step_map:
            raise ValueError(f"Unsupported splting parameter: {split_by}")

        step = step_map[split_by]
        self.time_intervalls = []

        current = self.start_time
        while current < self.end_time:
            next_time = current + step
            self.time_intervalls.append((current, min(next_time, self.end_time)))
            current = next_time
        print(f"we have the following time intervalls{self.time_intervalls}")

class Follower_Edge:
    #follows.tsv
    def __init__(self,edge_entry):
        self.id = edge_entry[0]
        self.time_scraped

class Truth:
    def __init__(self,truth_entry):
        try:
            try:
                time = pd.to_datetime(truth_entry[1], format='mixed', dayfirst=False,errors='coerce')
                self.timestamp = time.to_pydatetime()
            except:
                print(truth_entry[1])
                self.timestamp = datetime(9999, 12, 31, 23, 59, 59) 
            self.id = truth_entry[0]
            self.text = truth_entry[9]
            self.like_count= int(truth_entry[6])
            self.retruth_count= int(truth_entry[7])
            self.reply_count= int(truth_entry[8])
        except:
            print("there was an error reading the following truth id:")
            print(truth_entry)
class Node:
    #users.tsv
    def __init__(self ,entry,time_intervalls):
        entry= entry.split("\t")
        self.time_intervalls=time_intervalls
        self.truths_in_time_intervall=[]

        for _  in self.time_intervalls.time_intervalls:
            self.truths_in_time_intervall.append([])
        self.id=entry[0]
        self.timestamp=entry[1]
        self.time_scraped=entry[2]
        self.username=entry[3]
        self.follower_count=entry[4]
        self.following_count=entry[5]
        self.follower_edges=[]
        self.following_edges=[]
        self.truths={}
        #self.truths_at_timestemp={}

    def node_info(self):
        #print(f"Information about node {self.id}:")
        #print(f"Username: {self.username}")
        #print(f"follower count: {self.follower_count}")
        #print(f"following count: {self.following_count}")
        #print(f"number of truths:{len(self.truths)}")
        #counter=0
        #for truth_ids in self.truths_in_time_intervall:
        #    print(f"following nodes:{str(truth_ids)} are in the following time intervall:{str(self.time_intervalls.time_intervalls[counter])} ")
        #    counter+=1
        for name, value in vars(self).items():
            print(f"{name}: {value}")
        for truth_ids in self.truths_in_time_intervall:
            for truth_id in truth_ids:
                print(f"Details about truth id:{truth_id}")
                for name, value in vars(self.truths[truth_id]).items():
                     print(f"{name}: {value}")
        
        
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
            counter=0
            for intervall in self.time_intervalls.time_intervalls:
                if intervall[0] <= self.truths[truth_entry[0]].timestamp < intervall[1]: 
                    self.truths_in_time_intervall[counter].append(self.truths[truth_entry[0]].id)#TODO:this needs to be the truth id
                    #TODO: statement that catches -1 as timestamp 
                counter+=1

class TemporalGraph:
    def __init__(self,nodes_file="../Data/OriginalData/users.tsv",follower_file="../Data/OriginalData/follows.tsv",truths_file="../Data/OriginalData/truths.tsv",time_intervall_file="../Data/OriginalData/truths.tsv",emotion_csv="../Data/ProcessedData/enriched_comments_emotions.tsv"):
        """
        Docstring for __init__
        
        :param self: Description
        :param nodes_file: accepts either tsv or csv file of form #TODO:
        :param follower_file: accepts either tsv or csv file of form #TODO:
        :param truth_file: accepts either tsv or csv file of form #TODO:
        :param time_intervall_file: accepts either tsv or csv file of form #TODO:
        :param emotion_csv: accepts csv file with truth_id and emotion columns
        

        """
        self.nodes_dict={}
        self.time_intervall=self.create_timeintervall(time_intervall_file)
        self.create_nodes_from_file(nodes_file)
        self.create_follower_edges_from_file(follower_file)
        self.assign_truths_to_nodes(truths_file)
        self.assign_follower_edges()
        self.assign_emotion_to_truths(emotion_csv)
        self.assign_hate_to_truth(hate_csv="../Data/ProcessedData/truth_cleaned_enriched.csv")
        #self.assign_truth_to_truth(true_truth_csv="../Data/ProcessedData/truth_labels_prefilterd_gpt5.csv")
        self.active_nodes=[]

        #self.nodes_dict[self.active_nodes[1]].node_info()
            

        
    def add_node(self,node_entry):
        node = Node(node_entry,self.time_intervall)
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
                time_interval =TimeIntervall(lines.to_pydatetime())
                return time_interval
            
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

    def create_pyvis_representation(self):
        print("creating image of network...")
        Graph=nx.DiGraph()
        for node_id in tqdm.tqdm(list(self.nodes_dict.keys())):
            Graph.add_node(node_id)
        for node_id in tqdm.tqdm(list(self.nodes_dict.keys())):
            for following_node_id in list(self.nodes_dict[node_id].following_edges):
                Graph.add_edge(node_id,following_node_id,arrows="to")

        net = Network(directed=True)
        num_components = nx.number_connected_components(Graph)
        print("Number of connected components:", num_components)
        largest_cc = max(nx.connected_components(Graph), key=len)
        print("Largest component:", largest_cc)
        scc = list(nx.strongly_connected_components(Graph))
        print("Strongly connected components:", scc)
        net.from_nx(Graph)
        Graph.show("directed_graph.html")
        print("image created")
        print()

    def assign_emotion_to_truths(self,emotion_tsv):
        print("assigning emotions to truths ...")
        emotion_df=pd.read_csv(emotion_tsv,sep="\t")
        
        emotion_dict={}
        emotio_logits_dict={}
        for _, row in emotion_df.iterrows():
            emotion_dict[str(row['id'])]=row['emotion_label']
            emotio_logits_dict[str(row['id'])]=row['emotion_label_logits']
        #print(emotion_dict.keys())
        for node_id in tqdm.tqdm(list(self.nodes_dict.keys())):
            for truth_id in self.nodes_dict[node_id].truths.keys():
                try:

                    self.nodes_dict[node_id].truths[truth_id].emotion=emotion_dict[truth_id]
                except:
                    self.nodes_dict[node_id].truths[truth_id].emotion="unknown"
                try:
                    self.nodes_dict[node_id].truths[truth_id].emotion_logits=emotio_logits_dict[truth_id]
                except:
                    self.nodes_dict[node_id].truths[truth_id].emotion_logits="unknown"
        print("done.")

    def assign_hate_to_truth(self, hate_csv):
        print("assigning hate speech labels to truths ...")
        hate_df=pd.read_csv(hate_csv)
        hate_dict={}
        for _, row in hate_df.iterrows():
            hate_dict[str(row['id'])]=(row['hate_pred'],row['hate_prob'],row['sentiment_id'],row['sentiment_conf'],row['sentiment'],row['statement_flag'],row['statement_probability'],row["NO_STMT"],row["TRUE"],row["FALSE"])
        for node_id in tqdm.tqdm(list(self.nodes_dict.keys())):
            for truth_id in self.nodes_dict[node_id].truths:
                try:
                    self.nodes_dict[node_id].truths[truth_id].hate_speech_label=hate_dict[truth_id][0]
                except:
                    self.nodes_dict[node_id].truths[truth_id].hate_speech_label="unknown"
                try:
                    self.nodes_dict[node_id].truths[truth_id].hate_speech_prob=hate_dict[truth_id][1]
                except:
                    self.nodes_dict[node_id].truths[truth_id].hate_speech_prob="unknown"
                try:    
                    self.nodes_dict[node_id].truths[truth_id].sentiment_id=hate_dict[truth_id][2]
                except:
                    self.nodes_dict[node_id].truths[truth_id].sentiment_id="unknown"
                try:
                    self.nodes_dict[node_id].truths[truth_id].sentiment_conf=hate_dict[truth_id][3]
                except:
                    self.nodes_dict[node_id].truths[truth_id].sentiment_conf="unknown"  
                try:
                    self.nodes_dict[node_id].truths[truth_id].sentiment=hate_dict[truth_id][4]
                except:
                    self.nodes_dict[node_id].truths[truth_id].sentiment="unknown"
                try:
                    self.nodes_dict[node_id].truths[truth_id].statement_flag=hate_dict[truth_id][5]
                except:
                    self.nodes_dict[node_id].truths[truth_id].statement_flag="unknown"
                try:    
                    self.nodes_dict[node_id].truths[truth_id].statement_probability=hate_dict[truth_id][6]
                except:
                    self.nodes_dict[node_id].truths[truth_id].statement_probability="unknown"
                try:    
                    if hate_dict[truth_id][7] is not "":
                        self.nodes_dict[node_id].truths[truth_id].NO_STMT=hate_dict[truth_id][7]
                except:
                    self.nodes_dict[node_id].truths[truth_id].NO_STMT="unknown"
                try:
                    if hate_dict[truth_id][8] is not "":
                        self.nodes_dict[node_id].truths[truth_id].TRUE=hate_dict[truth_id][8]
                except:
                    self.nodes_dict[node_id].truths[truth_id].TRUE="unknown"        
                try:
                    if hate_dict[truth_id][9] is not "":
                        self.nodes_dict[node_id].truths[truth_id].FALSE=hate_dict[truth_id][9]  
                except:
                    self.nodes_dict[node_id].truths[truth_id].FALSE="unknown"
    
    #def assign_truth_to_truth(self, true_truth_csv):
    #    print("assigning truth to truth relations ...")
    #    truth_relation_df=pd.read_csv(true_truth_csv)
    #    true_truth_labels={}
    #    for index, row in truth_relation_df.iterrows():
    #        true_truth_labels[str(row['id'])]=str(row['probabilities'])
    #    for node_id in tqdm.tqdm(list(self.nodes_dict.keys())):
    #        for truth_id in self.nodes_dict[node_id].truths:
    #            try:
    #                self.nodes_dict[node_id].truths[truth_id].probabilites=true_truth_labels[truth_id]
    #            except:
    #                self.nodes_dict[node_id].truths[truth_id].probabilites="unknown"


    def find_truth(self,truth_id):
        for node_id in self.nodes_dict.keys():
            if truth_id in self.nodes_dict[node_id].truths.keys():
                for name, value in vars(self.nodes_dict[node_id].truths[truth_id]).items():
                     print(f"{name}: {value}")
        return None