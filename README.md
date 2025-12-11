# TrueGraphDynamics

This is a project work for the TUHH Module:
Deep Learning for Social Analytics

Team members:
- Johann Strunck
- Vincent Ridder-Schipkowski
- Sargunpreet Kaur
- Yunus Aras

## Short Description:

This project investigates the temporal evolution of user sentiment and perceived truthfulness in social networks using dynamic graph modeling. Leveraging data from the TrueSocial platform—comprising user comments, interactions, and timestamps—we construct time-resolved social graphs where nodes represent users enriched with features such as sentiment and truthfulness scores derived from natural language analysis, as well as interaction metrics (followers, likes, replies). Edges encode social relationships and interactions, allowing the network to evolve over discrete time intervals.

We employ Temporal Graph Neural Networks (TGNNs), to model and predict user-level dynamics, such as changes in sentiment, trustworthiness, and future social connections. The framework enables the analysis of influence patterns, identification of clusters of low-truth or highly influential users, and detection of opinion shifts or potential misinformation cascades. By integrating temporal and relational information, this study aims to uncover the mechanisms driving trust and sentiment propagation in online social ecosystems.

## Project: Temporal Trust & Sentiment Dynamics in TrueSocial Graph
## Steps to archive in the Project:

### Dataset

Source: [TrueSocial platform ](zenodo.org/records/7531625)
Features:

- Comments (text)

- Timestamp

- Social interactions (likes, subscriptions, replies, etc.)

Comment: depending on the development of the project datasets from Twitter and Reddit might be aquired to use for training and testing purposes.

### Step 0 — Data Preparation
- Define Node Feature Vector

    - Each user (node) will have features derived from comments and interactions.

- Analyze Graph Connectivity

    - Number of users

    - Active vs. inactive users

    - Interaction density

- Clean Dataset

    - Remove inactive users

    - Remove spam / low-information messages (optional)

    - convert emojis and hastags to usable tokens

- Standardize Timestamps for TGNN

    - Convert timestamps to a consistent format

    - Choose granularity (daily / weekly snapshots) of time intervals 

- Graph Reduction for Prototyping

    - Possible strategies:

        - Select a subgraph (e.g., the largest connected component, subgraph with high connectivity)

        - Limit to a specific topic or time window

### Step 1 — Data Enrichment
- Add Model-Generated Labels per Comment

    - Truthfulness Score (0–1):

        - Using LLM inference + prompt engineering

        - Using the network architecturen of following [paper](https://dl.acm.org/doi/abs/10.1145/3137597.3137600) 

    - Sentiment Classification

        - following the fings of [this paper](https://arxiv.org/abs/1901.04856)

        - decoding of sentiment vector into smaller feature vector 



### Step 2 — Temporal Graph Construction

For each timestamp period:

-   Nodes: users

    - Node information vector includes (all being time series data):

        - Sentiment from written posts 
            
        - Sentiment from liked posts 
            
        - Truthfulness score from written posts 

        - Truthfulness score from  liked posts 

        - Followers 

        - Following 
            
        - ...


- Edges:

    - Subscription / follower relationships

    - Reply or mention interactions

    - Graph evolves over time → Dynamic Graph Sequence

### Step 3 — Model Training

Approach: 

- Temporal Graph Neural Network (TGNN)

Example architectures:

- Temporal Graph Attention Networks

- Recurrent GNNs

- Dynamic Graph Convolutions

Goal: 

- Predict:
    - temporal evolution

    - Changes in user sentiment/truthfulness

    - Future connections (link prediction)


Training signal:

- Compare next timestamp graph vs. model prediction

### Step 4 — Model Evaluation & Analysis

Questions we aim to answer:

- Can the model predict future user state (sentiment & truthfulness)?

- How strongly do neighbors influence a user’s trust and sentiment?

- Do clusters of low-truth users emerge over time?

- Can we detect opinion shifts or misinformation cascades?

- can we identify highly influencial people in terms of truthfullness?

- is there a connection between the influence of a person on others and their sentiment?
