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

The regenerated [test_results.csv](test_results.csv) contains 20 customer
requests and 60 extracted line items.

### Request outcomes

- 6 requests were completely fulfilled.
- 11 requests were partially fulfilled.
- 3 requests had no fulfilled items.
- 12 of 20 requests contained at least one item requiring clarification.

### Item outcomes

| Status | Items | Share |
|---|---:|---:|
| Fulfilled | 34 | 56.7% |
| Cannot deliver on time | 12 | 20.0% |
| Needs clarification | 14 | 23.3% |

The fulfilled items generated 34 sales transactions and 51 supplier stock
transactions.

At the request dates shown in the first and final recorded rows:

- Cash changed from `$45,117.20` to `$45,380.90`, an increase of `$263.70`.
- Inventory value changed from `$4,875.30` to `$4,327.20`, a decrease of
  `$548.10`.
- Every response passed the built-in evaluation assertions with no parsing or
  arithmetic failures.

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

### Remaining delivery-date limitations

The regenerated run now separates:

- `request_date`, such as April 1
- A requested delivery deadline inside the customer text, often April 15

The 12 late items therefore represent genuine supplier lead-time limitations
relative to the requested deadline, rather than a comparison against the
request date.

### Pricing quality

All 34 fulfilled items used a valid numeric discount and passed subtotal and
total arithmetic assertions. The used rates were:

```text
0, 0.05, 0.1, 0.2
```

The arithmetic is deterministic, but the discount decision remains
LLM-dependent. A formal discount policy would make comparable orders more
consistent.

### Request extraction quality

The regenerated results preserve every CSV job and event value. There are no
`unknown` jobs caused by LLM extraction.

### Result serialization

`test_results.csv` now stores valid JSON in the response column, including its
evaluation metrics.

```python
"response": json.dumps(response)
```

## 5. Recommended Improvements

1. **Expand deterministic quote rules.** Define explicit quantity tiers for the
   discount decision while retaining the Quote Agent for recommendation and
   rationale:

   ```text
   total = unit_price * quantity * (1 - discount_rate)
   ```

   This guarantees that the quoted total always matches the stated discount.

2. **Add delivery-deadline metrics.** Track whether each item can meet the
   requested deadline and how many fulfillment days remain.

3. **Expand evaluation assertions.** The current runner serializes responses as
   JSON and validates exact product mappings, item statuses, numeric discount
   rates, expected arithmetic, transaction types, dates, and unique transaction
   IDs. Future work can add expected mapping and outcome fixtures for every CSV
   request.

4. **Add an approval stage for purchases.** Inventory and quotes can be
   generated automatically, but final purchase and sale transactions should be
   tied to explicit quote acceptance.

## Conclusion

The system successfully demonstrates agent specialization, structured
multi-item extraction, date-based data access, explanatory pricing, and atomic
fulfillment. Its main limitation is not agent coordination but the consistency
of business rules: delivery deadlines and price arithmetic should be
controlled deterministically rather than inferred from the LLM.
