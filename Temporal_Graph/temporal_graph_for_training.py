

from datetime import datetime, timedelta
from math import e
import time
from venv import create
import comm
from dateutil.relativedelta import relativedelta
from enum import unique
import pandas as pd
import regex as re
from pyvis.network import Network
import networkx as nx
import tqdm
import numpy as np
import ast

class TimeIntervall:
    def __init__(self,time_data,split_by="week"):
        
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
        self.generate_time_intervals(split_by)    

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

    
class Comment:
    def __init__(self, df_row):
        

        # keep your original behavior
        for col in df_row.index:
            setattr(self, col, df_row[col])

        # -------- helpers --------
        def parse(x):
            # already numeric
            if isinstance(x, (int, float, np.number)):
                return x

            # already array-like
            if isinstance(x, (list, tuple, np.ndarray)):
                return x

            # string case
            if isinstance(x, str):
                x = x.strip()

                # case 1: python-style list
                try:
                    return ast.literal_eval(x)
                except Exception:
                    pass

                # case 2: numpy-style vector "[0. 1. 0. 0.]"
                if x.startswith("[") and x.endswith("]"):
                    try:
                        return np.fromstring(x[1:-1], sep=" ")
                    except Exception:
                        pass

                # case 3: scalar in string form
                try:
                    return float(x)
                except Exception:
                    raise ValueError(f"Unparseable feature value: {x}")

            return x


        def to_array(x):
            x = parse(x)

            if isinstance(x, np.ndarray):
                return x.astype(np.float32)

            if isinstance(x, (list, tuple)):
                return np.asarray(x, dtype=np.float32)

            # scalar
            return np.array([x], dtype=np.float32)


        # -------- build feature vector --------
        parts = [
            to_array(self.bert_label_vec),
            to_array(self.text_feat),
            to_array(self.style_cluster_cats),
            to_array(self.topic_label_cats),
            to_array(self.sentiment_vec),
            to_array(self.activity_cluster_cats),
            to_array(self.temporal_rhythm_cluster_cats),
            to_array(self.hate_vec),
            to_array(self.statement_vec),
            to_array(self.like_count_scaled),
            to_array(self.retruth_count_scaled),
            to_array(self.reply_count_scaled),
            to_array(self.cluster_vec),
        ]

        self.feature_vec = torch.tensor(
            np.concatenate(parts, axis=0),
            dtype=torch.float
        )
        

class UserNode:
    def __init__(self,timeintervall, user_id):
        self.user_id = user_id
        self.comments = {}
        self.comments_in_intervall = [[] for _ in range(len(timeintervall.time_intervalls))]
        self.truth_flag_intervall = [None for _ in range(len(timeintervall.time_intervalls))]
        self.ingoing_edges = []
        self.outgoing_edges = []


class TemporalGraph:
    def __init__(self,follower_file="../Data/OriginalData/follows.tsv",comment_file="../Data/ProcessedData/network_modeling_data.csv",split_by="week"):
        self.comment_df = pd.read_csv(comment_file)
        self.time_intervalls = self.create_timeintervall()
        self.user_nodes = {}
        print("creating user nodes ...")
        self.create_user_nodes(self.comment_df)        
        print("assigning comments to intervalls ...")
        self.assign_comments_to_intervalls()
        print("creating edges ...")
        self.create_edges(pd.read_csv(follower_file, sep="\t"))
        print("assigning truth flags ...")
        self.assign_truth_flags()

    def create_timeintervall(self):
        print("creating time intervall ...")
        self.comment_df["timestamp"] = pd.to_datetime(self.comment_df["timestamp"], format='mixed', dayfirst=False,errors='coerce')
        self.time_intervall = TimeIntervall(self.comment_df["timestamp"], split_by="week")  

    def create_user_nodes(self,df):
        for _, row in df.iterrows():
            user_id = row["author"]
            if str(user_id) not in self.user_nodes.keys():
                self.user_nodes[str(user_id)] = UserNode(self.time_intervall, user_id)

    def create_edges(self, edge_df):
        user_nodes = self.user_nodes
        user_keys = set(user_nodes.keys())

        # vectorized cleanup
        sources = (
            edge_df["follower"]
            .astype(str)
            .str.replace(r"[^0-9.]", "", regex=True)
            .values
        )
        targets = (
            edge_df["followed"]
            .astype(str)
            .str.replace(r"[^0-9.]", "", regex=True)
            .values
        )

        counter = len(sources)
        correct_edges_counter = 0

        for src, tgt in zip(sources, targets):
            if src in user_keys and tgt in user_keys:
                correct_edges_counter += 1
                user_nodes[src].outgoing_edges.append(user_nodes[tgt])
                user_nodes[tgt].ingoing_edges.append(user_nodes[src])

        print(f"of {counter} edges, {correct_edges_counter} edges are correct")

    
    def assign_comments_to_intervalls(self):
        for _, row in self.comment_df.iterrows():
            comment = Comment(row)
            user_id = str(comment.author)
            if user_id in self.user_nodes.keys():
                user_node = self.user_nodes[user_id]
                user_node.comments[comment.id] = comment
                for i, (start, end) in enumerate(self.time_intervall.time_intervalls):
                    if start <= comment.timestamp < end:
                        user_node.comments_in_intervall[i].append(comment)
                        break
            else:
                print(f"Warning: Comment with user_id {user_id} has no corresponding user node!")

    def assign_truth_flags(self):
        for user_id, user_node in self.user_nodes.items():
            for i in range(len(user_node.comments_in_intervall)):

                window_comments = []
                for j in range(max(0, i - 2), i + 1):
                    window_comments.extend(user_node.comments_in_intervall[j])

                true_count = sum(
                    comment.gpt_label == "TRUE"
                    for comment in window_comments
                )

                # ---- SOFT LABEL (0.0 to 1.0) ----
                user_node.truth_flag_intervall[i] = min(true_count / 3.0,1.0)


import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------------
# CONFIG
# -----------------------------------
USE_PAST_Y = True
COMMENT_HIDDEN = 128
USER_HIDDEN = 128


# =====================================================
# COMMENT ENCODER
# =====================================================
class CommentEncoder(nn.Module):
    def __init__(self, comment_dim, hidden_dim):
        super().__init__()
        self.gru = nn.GRU(comment_dim, hidden_dim, batch_first=True)

    def forward(self, comments):

        if len(comments) == 0:
            return None

        x = torch.stack([c.feature_vec for c in comments]).to(DEVICE)
        _, h = self.gru(x.unsqueeze(0))
        return h.squeeze(0)


# =====================================================
# USER TEMPORAL ENCODER
# =====================================================
class UserTemporalEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.gru = nn.GRUCell(in_dim, hidden_dim)

    def forward(self, x_t, h_prev):
        return self.gru(x_t, h_prev)


# =====================================================
# GRAPH SAGE (Mean Aggregation)
# =====================================================
class GraphSAGEAggregator(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.self_lin = nn.Linear(hidden_dim, hidden_dim)
        self.neigh_lin = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, user_states, temporal_graph):

        new_states = {}

        for u_id, u_node in temporal_graph.user_nodes.items():

            self_msg = self.self_lin(user_states[u_id])

            neigh_states = [
                user_states[str(v.user_id)]
                for v in u_node.outgoing_edges
            ]

            if len(neigh_states) > 0:
                neigh_msg = torch.mean(
                    torch.stack(neigh_states), dim=0
                )
                neigh_msg = self.neigh_lin(neigh_msg)
            else:
                neigh_msg = torch.zeros_like(self_msg)

            new_states[u_id] = torch.relu(self_msg + neigh_msg)

        return new_states


# =====================================================
# GRAPH ATTENTION NETWORK (Single-Head)
# =====================================================
class GraphGATAggregator(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn = nn.Linear(2 * hidden_dim, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, user_states, temporal_graph):

        new_states = {}

        # Linear transform for all nodes
        Wh = {
            u: self.W(h)
            for u, h in user_states.items()
        }

        for u_id, u_node in temporal_graph.user_nodes.items():

            h_u = Wh[u_id]

            neigh_ids = [
                str(v.user_id)
                for v in u_node.outgoing_edges
            ]

            if len(neigh_ids) == 0:
                new_states[u_id] = torch.relu(h_u)
                continue

            attn_scores = []
            neigh_feats = []

            for v_id in neigh_ids:

                h_v = Wh[v_id]
                concat = torch.cat([h_u, h_v], dim=0)
                e_uv = self.leaky_relu(self.attn(concat))

                attn_scores.append(e_uv)
                neigh_feats.append(h_v)

            attn_scores = torch.stack(attn_scores)      # (N,1)
            alpha = torch.softmax(attn_scores, dim=0)   # attention weights

            neigh_feats = torch.stack(neigh_feats)      # (N, hidden)
            h_neigh = torch.sum(alpha * neigh_feats, dim=0)

            new_states[u_id] = torch.relu(h_neigh)

        return new_states


# =====================================================
# FULL TEMPORAL MODEL
# =====================================================
class TemporalTruthModel(nn.Module):
    def __init__(
        self,
        comment_dim,
        use_past_y=True,
        use_graph=True,
        use_comments=True,
        graph_type="sage"   # <---- NEW SWITCH
    ):
        super().__init__()

        self.use_past_y = use_past_y
        self.use_graph = use_graph
        self.use_comments = use_comments
        self.graph_type = graph_type

        past_y_dim = 1 if use_past_y else 0

        self.comment_encoder = CommentEncoder(
            comment_dim, COMMENT_HIDDEN
        )

        self.user_encoder = UserTemporalEncoder(
            COMMENT_HIDDEN + past_y_dim,
            USER_HIDDEN
        )

        # -------- Graph Layer Selection --------
        if graph_type == "sage":
            self.graph_agg = GraphSAGEAggregator(USER_HIDDEN)
        elif graph_type == "gat":
            self.graph_agg = GraphGATAggregator(USER_HIDDEN)
        else:
            raise ValueError("graph_type must be 'sage' or 'gat'")

        self.classifier = nn.Linear(USER_HIDDEN, 1)


    # =================================================
    # Forward Full Sequence
    # =================================================
    def forward(self, temporal_graph):

        T = len(temporal_graph.time_intervall.time_intervalls)
        user_ids = list(temporal_graph.user_nodes.keys())

        user_states = {
            u: torch.zeros(USER_HIDDEN, device=DEVICE)
            for u in user_ids
        }

        all_logits = []
        all_labels = []

        for t in range(T):

            for u in user_ids:

                node = temporal_graph.user_nodes[u]
                comments = node.comments_in_intervall[t]

                # ---- Comment encoder ----
                if self.use_comments:
                    h_c = self.comment_encoder(comments)
                    if h_c is None:
                        h_c = torch.zeros(COMMENT_HIDDEN, device=DEVICE)
                    else:
                        h_c = h_c.view(-1)
                else:
                    h_c = torch.zeros(COMMENT_HIDDEN, device=DEVICE)

                # ---- Past y ----
                if self.use_past_y:
                    prev_y = node.truth_flag_intervall[t-1] if t > 0 else 0.0
                    prev_y_tensor = torch.tensor(
                        [prev_y], device=DEVICE, dtype=torch.float
                    )
                    x_t = torch.cat([h_c, prev_y_tensor])
                else:
                    x_t = h_c

                user_states[u] = self.user_encoder(
                    x_t, user_states[u]
                )

            # ---- Graph Layer ----
            if self.use_graph:
                user_states = self.graph_agg(
                    user_states, temporal_graph
                )

            logits = torch.stack([
                self.classifier(user_states[u]).squeeze()
                for u in user_ids
            ])

            labels = torch.tensor(
                [
                    temporal_graph.user_nodes[u]
                    .truth_flag_intervall[t]
                    for u in user_ids
                ],
                dtype=torch.float,
                device=DEVICE
            )

            all_logits.append(logits)
            all_labels.append(labels)

        return all_logits, all_labels


    # =================================================
    # One Step (used in training)
    # =================================================
    def forward_one_step(self, temporal_graph, t, user_states):

        user_ids = list(temporal_graph.user_nodes.keys())

        for u in user_ids:

            node = temporal_graph.user_nodes[u]
            comments = node.comments_in_intervall[t]

            if self.use_comments:
                h_c = self.comment_encoder(comments)
                if h_c is None:
                    h_c = torch.zeros(COMMENT_HIDDEN, device=DEVICE)
                else:
                    h_c = h_c.view(-1)
            else:
                h_c = torch.zeros(COMMENT_HIDDEN, device=DEVICE)

            if self.use_past_y:
                prev_y = node.truth_flag_intervall[t-1] if t > 0 else 0.0
                prev_y_tensor = torch.tensor(
                    [prev_y], device=DEVICE, dtype=torch.float
                )
                x_t = torch.cat([h_c, prev_y_tensor])
            else:
                x_t = h_c

            user_states[u] = self.user_encoder(
                x_t, user_states[u]
            )

        if self.use_graph:
            user_states = self.graph_agg(
                user_states, temporal_graph
            )

        logits = torch.stack([
            self.classifier(user_states[u]).squeeze()
            for u in user_ids
        ])

        labels = torch.tensor(
            [
                temporal_graph.user_nodes[u]
                .truth_flag_intervall[t]
                for u in user_ids
            ],
            dtype=torch.float,
            device=DEVICE
        )

        return logits, labels, user_states
