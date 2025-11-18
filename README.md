# TrueGraphDynamics

This is a project work for the TuHH Model:
Deep Learning for Social Analytics

Working on this Project:
-Johan Strunk
-Vincent Ridder-Schipkowski
## Short Description:
Temporal Graph Neural Network project analyzing how sentiment and truthfulness evolve in a social network over time. Uses enriched user interaction data to predict future user states and network structure changes on the TrueSocial platform.

## Steps to archive in the Project:

🧠 Project: Temporal Trust & Sentiment Dynamics in TrueSocial Graph
📌 Dataset

Source: TrueSocial platform
Features:

Comments (text)

Timestamp

Social interactions (likes, subscriptions, replies, etc.)

⚙️ Step 0 — Data Preparation
🔹 Define Node Feature Vector

Each user (node) will have features derived from comments and interactions.

🔹 Analyze Graph Connectivity

Number of users

Active vs. inactive users

Interaction density

🔹 Clean Dataset

Remove inactive users

Remove spam / low-information messages (optional)

🔹 Standardize Timestamps

Convert timestamps to a consistent format

Choose granularity (daily / weekly snapshots)

🔹 Graph Reduction for Prototyping

Possible strategies:

Select a subgraph (e.g., the largest connected component)

Limit to a specific topic or time window

🧩 Step 1 — Data Enrichment
Add Model-Generated Labels per Comment

Using LLM inference + prompt engineering:

Truthfulness Score (0–1)

Sentiment Score (e.g., -1 negative → +1 positive)

User-Level Aggregation

For each user:

Average sentiment

Trust/truthfulness reputation score

Weighted by interactions (likes, reposts, replies)

🕸️ Step 2 — Temporal Graph Construction

For each timestamp period:

Nodes: users
Node information vector includes:

Sentiment history (time series)

Truthfulness reputation

Degree + engagement features

Edges:

Subscription / follower relationships

Reply or mention interactions

Likes (optional weighted edges)

Graph evolves over time → Dynamic Graph Sequence

🧪 Step 3 — Model Training

Approach: Temporal Graph Neural Network (TGNN)
Example architectures:

Temporal Graph Attention Networks

Recurrent GNNs

Dynamic Graph Convolutions

Goal: Predict temporal evolution

Changes in user sentiment/truthfulness

Future connections (link prediction)

Training signal:

Compare next timestamp graph vs. model prediction

📊 Step 4 — Model Evaluation & Analysis

Questions we aim to answer:

Can the model predict future user state (sentiment & truthfulness)?

How strongly do neighbors influence a user’s trust and sentiment?

Do clusters of low-truth users emerge over time?

Can we detect opinion shifts or misinformation cascades?