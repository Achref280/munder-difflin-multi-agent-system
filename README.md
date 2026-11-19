# Munder Difflin Multi-Agent Paper Company

A multi-agent workflow that processes paper-supply requests, checks dated
inventory, generates explainable quotes, and records fulfillment transactions.

## Overview

The project uses five `smolagents` agents:

1. **Orchestration Agent** coordinates each request line item.
2. **Request Processor Agent** maps natural-language products to exact
   catalog items and extracts the requested delivery deadline.
3. **Inventory Agent** checks stock, shortages, and supplier delivery dates.
4. **Quote Agent** recommends a numeric bulk discount and explains its
   decision.
5. **Order Agent** calls one atomic fulfillment tool.

Python validates the LLM results and performs business-critical calculations.
An unsupported item blocks only that item; valid items in the same request can
continue through the workflow.

See the [workflow diagram](flowchart.md) and [evaluation report](EVALUATION_REPORT.md).

## Features

- Multi-item natural-language request processing
- Exact catalog validation with per-item clarification
- Trusted request date, customer job, and event metadata from scenario CSV data
- Separate request date, requested delivery date, and supplier delivery date
- Date-based inventory, cash, and historical quote lookups
- Structured `final_answer` schemas for request, quote, and order agents
- Deterministic price arithmetic:

  ```text
  subtotal = unit_price × quantity
  total = subtotal × (1 - discount_rate)
  ```

- Separate shortage and safety-stock supplier purchases
- Atomic shortage, safety-stock, and sale transaction execution
- Automatic inventory initialization for valid catalog items not included in
  the sampled inventory
- JSON response serialization and embedded evaluation assertions
- Unit tests plus opt-in real-LLM integration tests

## Project Structure

```text
.
├── project_starter.py          # Agents, tools, database logic, and evaluation
├── project_starter_AI.py       # Alternate deterministic AI implementation
├── tests/
│   └── test_workflow.py        # Unit and integration tests
├── quote_requests.csv          # Historical requests
├── quote_requests_sample.csv   # Scenario evaluation requests
├── quotes.csv                  # Historical quote data
├── munder_difflin.db           # SQLite project database
├── test_results.csv            # Generated evaluation results
├── flowchart.md                # Mermaid workflow diagram
├── EVALUATION_REPORT.md        # Architecture and evaluation analysis
└── requirements.txt
```

## Requirements

- Python 3.10 or later is recommended.
- An OpenAI-compatible API key available to the model client.
- The project uses the endpoint:

  ```text
  https://openai.vocareum.com/v1
  ```

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
UDACITY_OPENAI_API_KEY=your_api_key
```

Do not commit `.env` or API keys.

## Run the Evaluation

From the repository root:

```bash
python project_starter.py
```

The scenario runner:

1. Reinitializes `munder_difflin.db`.
2. Loads and sorts `quote_requests_sample.csv`.
3. Processes every request through the agents.
4. Prints cash, inventory, and per-item responses.
5. Validates each completed workflow response.
6. Writes `test_results.csv`.

Each response stores:

- Request and requested delivery dates
- Trusted job and event metadata
- Per-item status
- Inventory and delivery result
- Quote, discount, subtotal, total, and pricing explanation
- Transaction details
- Evaluation assertions

The response column in `test_results.csv` is valid JSON.

## Run Tests

Run the deterministic test suite:

```bash
python -m unittest tests/test_workflow.py
```

Run the optional real-LLM integration tests:

```bash
export RUN_LLM_INTEGRATION_TESTS=1
python -m unittest tests/test_workflow.py
```

The integration tests require `UDACITY_OPENAI_API_KEY`.

## Workflow Behavior

For each item:

```text
Catalog validation
  -> Clarification required? stop this item
  -> Inventory check at request_date
  -> Can shortage arrive by requested_delivery_date?
       -> No: stop this item
       -> Yes: quote with deterministic pricing
  -> Atomic order execution
```

For an order with insufficient stock:

```text
Purchase shortage quantity
Purchase safety-stock quantity
Record customer sale
```

The two supplier purchases use independent delivery dates, because including
safety stock can change the supplier lead-time bracket.

## Safety and Validation

- LLM-selected product names must exist in `paper_supplies`.
- Invalid quantities and dates are rejected.
- Discounts must be numeric values between `0` and `1`.
- Price totals are calculated in Python, not trusted from the LLM.
- Fulfillment writes are executed in one SQLite transaction.
- Evaluator assertions verify:
  - Status values
  - Exact product mappings
  - Pricing arithmetic
  - Transaction types and dates
  - Integer and unique transaction IDs

## Current Evaluation

The regenerated 20-request evaluation produced:

- 60 line items
- 34 fulfilled items
- 12 items unable to meet the requested deadline
- 14 items requiring clarification
- 34 sales transactions
- 51 stock-order transactions
- 20 responses passing embedded evaluation assertions

See [EVALUATION_REPORT.md](EVALUATION_REPORT.md) for details.

## Known Limitations

- LLM discount recommendations are not yet governed by formal quantity tiers.
- A quote-acceptance step is not currently required before fulfillment.
- Catalog variants such as A3 products and specialty colors may require
  clarification.
- Real-LLM tests depend on external availability, latency, and pricing.
- The sample financial model is educational rather than production accounting.

## Future Improvements

1. Add explicit approval before purchase and sale transactions.
2. Define deterministic discount tiers while retaining LLM explanations.
3. Add expected outcomes for every scenario row.
4. Expand catalog variants for common A3/A5 and product specifications.
5. Track processing cost and latency per agent/request.
6. Add a customer simulator that uses the scenario role, mood, event, and
   request to answer clarification questions and negotiate acceptance.
7. Add a terminal visualization that renders structured agent events with
   progress indicators; presentation should remain separate from business
   logic.
8. Add read-only business-advisor capabilities that analyze completed
   transactions and recommend inventory, supplier, pricing, and catalog
   improvements.

The current educational implementation uses five agents. A future production
version is not required to keep that limit and can introduce dedicated
Customer and Business Advisor agents. The customer simulator can also remain
deterministic if no additional LLM is needed for the demonstration.

An enhanced demonstration workflow would be:

```text
Customer simulator
  -> Request negotiation
  -> Five-agent operational workflow
  -> Structured progress events
  -> Terminal visualization
  -> Completed transactions
  -> Business-advisor recommendations
```

## License

This repository is an educational project.
