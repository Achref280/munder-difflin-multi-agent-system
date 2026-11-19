# Munder Difflin Multi-Agent System Report

## 1. System Overview

The project is a five-agent paper-order workflow built with `smolagents`,
SQLite, SQLAlchemy, and Pandas:

1. **Orchestration Agent** coordinates each request line item.
2. **Request Processor Agent** extracts multiple products, quantities, dates,
   customer context, and catalog matches.
3. **Inventory Agent** checks date-based stock, shortages, and delivery timing.
4. **Quote Agent** uses catalog prices, historical quotes, cash, inventory, and
   financial data to calculate a price and explain the decision.
5. **Order Agent** executes fulfillment through one atomic `execute_order` tool.

The request processor is constrained with a structured `final_answer` schema.
Python then validates exact catalog names, positive quantities, dates, and
candidate products. Validation is intentionally per item: an ambiguous item
is returned for clarification without blocking valid items in the same request.

For each valid item, the workflow is:

```text
Extract request
  -> Validate catalog mapping
  -> Check stock and supplier delivery date
  -> Stop this item if it cannot arrive on time
  -> Generate quote and pricing explanation
  -> Execute transactions atomically
```

`execute_order` writes all required changes in one database transaction. If
stock is short, it can create:

1. One stock purchase for the shortage.
2. A separate stock purchase for safety stock, because its delivery lead time
   can differ.
3. One customer sale transaction.

Catalog items that were not included in the randomly generated inventory can
still be ordered. They receive zero initial stock and the configured default
safety-stock level.

## 2. Evaluation Results

The evaluated [test_results.csv](test_results.csv) contains 20 customer requests
and 60 extracted line items.

### Request outcomes

- 1 request was completely fulfilled.
- 7 requests were partially fulfilled.
- 12 requests had no fulfilled items.
- 12 of 20 requests contained at least one item requiring clarification.

### Item outcomes

| Status | Items | Share |
|---|---:|---:|
| Fulfilled | 10 | 16.7% |
| Cannot deliver on time | 37 | 61.7% |
| Needs clarification | 13 | 21.7% |

The fulfilled items generated 10 sales transactions and 7 supplier stock
transactions.

At the request dates shown in the first and final recorded rows:

- Cash changed from `$45,092.20` to `$45,573.60`, an increase of `$481.40`.
- Inventory value changed from `$4,905.30` to `$4,436.40`, a decrease of
  `$468.90`.

## 3. Strengths

- **Multi-item processing:** one request can contain several independently
  processed products.
- **Per-item failure isolation:** clarification or lateness for one product
  does not prevent valid products from continuing.
- **Date-aware stock:** inventory, cash, and historical quotes are queried at a
  specific request date.
- **Explainable pricing:** fulfilled quotes include unit price, subtotal,
  discount rationale, and final total.
- **Transaction safety:** shortage, safety-stock, and sale transactions are
  committed together by `execute_order`.
- **Catalog-only fulfillment:** a valid catalog item can be ordered even if the
  sampled inventory did not initially include it.
- **Test coverage:** 34 deterministic workflow tests pass, with 4 optional
  real-LLM integration tests available separately.

## 4. Areas for Improvement

### Delivery-date interpretation in the evaluated run

The evaluated `test_results.csv` was generated before the request date and
delivery deadline were separated. The scenario file contains:

- `request_date`, such as April 1
- A requested delivery deadline inside the customer text, often April 15

That run used `request_date` as both the inventory snapshot date and the
fulfillment deadline. This is the main reason 37 of 60 historical items were
reported as too late. The current code now distinguishes:

```text
request_date
requested_delivery_date
supplier_delivery_date
```

### Pricing consistency in the evaluated run

Ten historical items were priced, but four totals did not agree with the
stated discount
and catalog subtotal. Discount output also mixed representations such as:

```text
0.1
10%
10
"10% bulk discount applied"
```

The current implementation now requires a numeric discount rate from the Quote
Agent and calculates unit totals in deterministic Python.

### Request extraction quality

Eight of 20 historical results contained `job: "unknown"`, while several
others stored phrases such as “place an order” rather than a customer role.
The current runner now passes CSV `job` and `event` values as trusted process
inputs and uses the LLM only for products and delivery-deadline extraction.

### Result serialization

`test_results.csv` currently stores Python dictionary text. JSON serialization
would make the evaluation output easier to parse and reuse:

```python
"response": json.dumps(response)
```

## 5. Recommended Improvements

1. **Regenerate the evaluation.** Run the updated workflow to measure the effect
   of delivery-date separation, deterministic pricing, and trusted metadata.

2. **Expand deterministic quote rules.** Define explicit quantity tiers for the
   discount decision while retaining the Quote Agent for recommendation and
   rationale:

   ```text
   total = unit_price * quantity * (1 - discount_rate)
   ```

   This guarantees that the quoted total always matches the stated discount.

3. **Add delivery-deadline metrics.** Track whether each item can meet the
   requested deadline and how many fulfillment days remain.

4. **Expand evaluation assertions.** The current runner serializes responses as
   JSON and validates exact product mappings, item statuses, numeric discount
   rates, expected arithmetic, transaction types, dates, and unique transaction
   IDs. Future work can add expected mapping and outcome fixtures for every CSV
   request.

5. **Add an approval stage for purchases.** Inventory and quotes can be
   generated automatically, but final purchase and sale transactions should be
   tied to explicit quote acceptance.

## Conclusion

The system successfully demonstrates agent specialization, structured
multi-item extraction, date-based data access, explanatory pricing, and atomic
fulfillment. Its main limitation is not agent coordination but the consistency
of business rules: delivery deadlines and price arithmetic should be
controlled deterministically rather than inferred from the LLM.
