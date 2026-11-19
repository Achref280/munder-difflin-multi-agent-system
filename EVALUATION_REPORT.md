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
- **Test coverage:** 32 deterministic workflow tests pass, with 4 optional
  real-LLM integration tests available separately.

## 4. Areas for Improvement

### Delivery-date interpretation

The scenario file contains:

- `request_date`, such as April 1
- A requested delivery deadline inside the customer text, often April 15

The current workflow uses `request_date` as both the inventory snapshot date
and the fulfillment deadline. This is the main reason 37 of 60 items were
reported as too late. The system should distinguish:

```text
request_date
requested_delivery_date
supplier_delivery_date
```

### Pricing consistency

Ten items were priced, but four totals did not agree with the stated discount
and catalog subtotal. Discount output also mixed representations such as:

```text
0.1
10%
10
"10% bulk discount applied"
```

The explanation is useful, but the arithmetic is currently still dependent on
the LLM.

### Request extraction quality

Eight of 20 top-level results contained `job: "unknown"`, while several other
results stored phrases such as “place an order” rather than a customer role.
The runner already supplies reliable job and event metadata, so that context
should be passed directly into extraction or used to override incorrect LLM
extraction.

### Result serialization

`test_results.csv` currently stores Python dictionary text. JSON serialization
would make the evaluation output easier to parse and reuse:

```python
"response": json.dumps(response)
```

## 5. Recommended Improvements

1. **Separate request date from delivery deadline.** Extract the requested
   delivery date, use `request_date` for historical stock/finance checks, and
   compare supplier delivery against the actual deadline.

2. **Move price arithmetic into deterministic Python.** Let the Quote Agent
   recommend and explain a validated numeric discount, then calculate:

   ```text
   total = unit_price * quantity * (1 - discount_rate)
   ```

   This guarantees that the quoted total always matches the stated discount.

3. **Use the CSV job and event fields as trusted metadata.** Pass these values
   explicitly to `process_request` and reserve LLM extraction for product and
   deadline information.

4. **Add evaluation assertions and JSON output.** Record item status, exact
   product mapping, discount rate, expected arithmetic, and transaction IDs as
   structured JSON. This would allow the CSV to be evaluated automatically.

5. **Add an approval stage for purchases.** Inventory and quotes can be
   generated automatically, but final purchase and sale transactions should be
   tied to explicit quote acceptance.

## Conclusion

The system successfully demonstrates agent specialization, structured
multi-item extraction, date-based data access, explanatory pricing, and atomic
fulfillment. Its main limitation is not agent coordination but the consistency
of business rules: delivery deadlines and price arithmetic should be
controlled deterministically rather than inferred from the LLM.
