## Title

The Morpheus Protocol: Testing AI Agent(s) with AI agent(s)

## Description

Can AI agent(s) persuade another (agent/team of agents) with possibly rag to execute actions outside its intended scope?

## Project Description

I built a protocol designed to test how easily different AI models combined with various system prompts, RAG configurations, and agent setups can be convinced or manipulated. The goal is to give companies a way to thoroughly evaluate their AI agents/systems before deployment. Functioning similarly to a fuzzer that checks for weaknesses across a range of custom scenarios.

The system is highly flexible and supports:
- Custom RAG data ablations  
- Any number of sub-agents on both the "convincing" and "non-convincing" sides  
- Looping agents and multiple model types  
- Custom names, descriptions, prompts, and output keys  
- Optional unique tools  
- Full observability, including model errors, tool calls, metrics, and conversation history  

The technology stack includes google-adk[a2a] v1.18, LanceDB, and several formatting and parsing packages.

All agents must be defined inside the codebase, and the system does not connect to external services. It supports at least one remote agent lead and one non-remote agent lead. Plugins provide observability for both types, along with custom RAG tools to evaluate whether an agent can be persuaded or whether it resists persuasion. The framework is entirely customizable to simulate a wide variety of testing environments.

All outputs are produced in **JSON** to ensure the gold standard. Every function and class includes fully documented arguments. RAG queries are cached using an **LRU** cache to avoid concurrency issues and save on resources for each tool call.

The workflow begins with a custom YAML configuration file. Using the factory pattern, the system constructs all components with dependency injection where possible. After loading configs and optional RAG example data, the initial prompt is sent to the non-remote team, context is assembled, and output keys are prepared.

Execution continues until custom tools such as: `join_blue_team`, `take_red_pill`, or `take_blue_pill` are invoked. Or, if loops are enabled, until a maximum number of iterations is reached. All activity is logged into dedicated files (`app.log`, `debug.log`, `dialogue.log`, `error.log`, `metrics.log`) for inspection.

The codebase also includes prebuilt ablations to help users quickly understand the system and begin experimenting with custom scenarios.

The RAG system uses Google Gemini embedding model (`models/embedding-001`) to generate vector embeddings for all documents and queries. These embeddings are stored in LanceDB, which uses an IVF_PQ (Inverted File with Product Quantization) vector index for approximate nearest neighbor search once the table has at least 256 records. If fewer than 256 records exist, the system automatically falls back to a flat index until IVF_PQ becomes viable.

The agent tools provide built in security. This includes but not limited to, capping limit values, stopping empty/long queries, pattern matching to avoid rag tampering

There exists two template ablations: A matrix characters that convince a park ranger and one with a system admin without the ability for proofs to convince it to run the reboot "take_red_pill" (The Cassandra complex)


## Installing dependencies

```bash
# Install dependencies
pip install -e .
```

## Running

```bash
python3 main.py --config ablations/ablation-reality-check/ablation-reality-check.yaml 
```


## Images

![Red pill Example](images/example-ending-red-pill.jpg)

![Conversation Example](images/example-middle-conversation.jpg)

![Conversation Example](images/example-ending-stalemate.jpg)

![Cassandra Ablation](images/example-ablation-cassandra.jpg)

![Reality Check Ablation](images/example-ablation-reality-check.jpg)


## Further reading:

https://cloud.google.com/blog/topics/developers-practitioners/building-collaborative-ai-a-developers-guide-to-multi-agent-systems-with-adk

https://google.github.io/adk-docs/a2a/quickstart-consuming-go/

https://www.kaggle.com/competitions/agents-intensive-capstone-project