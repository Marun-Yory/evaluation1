# Evaluation1

This module covers systematic evaluation for search, RAG, and agent
systems.

We generate ground truth data with an LLM. Then we measure performance
with Hit Rate, MRR, and LLM-as-a-judge.


## Part 1: Search Evaluation

Part 1 creates a ground truth dataset and uses it to evaluate retrieval
quality.

1. Intro - Why evaluation matters, offline vs online
2. Generating Ground Truth - Structured output for one document
3. Generating Ground Truth for All Documents - Batch generation, cost, and prepared data
4. Search Evaluation - Search setup and relevance lists
5. Search Evaluation Metrics - Hit Rate, MRR, the evaluate() function
6. Search Parameter Tuning - Using metrics to tune boost values


## Part 2: RAG and Agent Evaluation

Part 2 evaluates answer quality after retrieval. It also shows the
basic idea of agent evaluation: save the final answer and the tool-call
trajectory.

1. RAG and Agent Evaluation - What changes after retrieval
2. Generating RAG Answers - Running RAG on the ground truth questions
3. LLM as a Judge - Using an LLM to evaluate answer quality
4. Agent Evaluation - Capturing answers and tool-call trajectories
5. Next Steps - Evaluation frameworks, monitoring, and resources
