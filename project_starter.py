import ast
import json
import os
import re
import time
from datetime import datetime, timedelta
from typing import Any

import dotenv
import numpy as np
import pandas as pd
from smolagents import (
    OpenAIServerModel,
    Tool,
    ToolCallingAgent,
    tool,
)
from sqlalchemy import Engine, create_engine
from sqlalchemy.sql import text

from event_emitter import emitter
from terminal_visualizer import TerminalVisualizer
from workflow_events import WorkflowEvent

# Create an SQLite database
db_engine = create_engine("sqlite:///munder_difflin.db")

# List containing the different kinds of papers
paper_supplies = [
    # Paper Types (priced per sheet unless specified)
    {"item_name": "A4 paper", "category": "paper", "unit_price": 0.05},
    {"item_name": "Letter-sized paper", "category": "paper", "unit_price": 0.06},
    {"item_name": "Cardstock", "category": "paper", "unit_price": 0.15},
    {"item_name": "Colored paper", "category": "paper", "unit_price": 0.10},
    {"item_name": "Glossy paper", "category": "paper", "unit_price": 0.20},
    {"item_name": "Matte paper", "category": "paper", "unit_price": 0.18},
    {"item_name": "Recycled paper", "category": "paper", "unit_price": 0.08},
    {"item_name": "Eco-friendly paper", "category": "paper", "unit_price": 0.12},
    {"item_name": "Poster paper", "category": "paper", "unit_price": 0.25},
    {"item_name": "Banner paper", "category": "paper", "unit_price": 0.30},
    {"item_name": "Kraft paper", "category": "paper", "unit_price": 0.10},
    {"item_name": "Construction paper", "category": "paper", "unit_price": 0.07},
    {"item_name": "Wrapping paper", "category": "paper", "unit_price": 0.15},
    {"item_name": "Glitter paper", "category": "paper", "unit_price": 0.22},
    {"item_name": "Decorative paper", "category": "paper", "unit_price": 0.18},
    {"item_name": "Letterhead paper", "category": "paper", "unit_price": 0.12},
    {"item_name": "Legal-size paper", "category": "paper", "unit_price": 0.08},
    {"item_name": "Crepe paper", "category": "paper", "unit_price": 0.05},
    {"item_name": "Photo paper", "category": "paper", "unit_price": 0.25},
    {"item_name": "Uncoated paper", "category": "paper", "unit_price": 0.06},
    {"item_name": "Butcher paper", "category": "paper", "unit_price": 0.10},
    {"item_name": "Heavyweight paper", "category": "paper", "unit_price": 0.20},
    {"item_name": "Standard copy paper", "category": "paper", "unit_price": 0.04},
    {"item_name": "Bright-colored paper", "category": "paper", "unit_price": 0.12},
    {"item_name": "Patterned paper", "category": "paper", "unit_price": 0.15},
    # Product Types (priced per unit)
    {
        "item_name": "Paper plates",
        "category": "product",
        "unit_price": 0.10,
    },  # per plate
    {"item_name": "Paper cups", "category": "product", "unit_price": 0.08},  # per cup
    {
        "item_name": "Paper napkins",
        "category": "product",
        "unit_price": 0.02,
    },  # per napkin
    {
        "item_name": "Disposable cups",
        "category": "product",
        "unit_price": 0.10,
    },  # per cup
    {
        "item_name": "Table covers",
        "category": "product",
        "unit_price": 1.50,
    },  # per cover
    {
        "item_name": "Envelopes",
        "category": "product",
        "unit_price": 0.05,
    },  # per envelope
    {
        "item_name": "Sticky notes",
        "category": "product",
        "unit_price": 0.03,
    },  # per sheet
    {"item_name": "Notepads", "category": "product", "unit_price": 2.00},  # per pad
    {
        "item_name": "Invitation cards",
        "category": "product",
        "unit_price": 0.50,
    },  # per card
    {"item_name": "Flyers", "category": "product", "unit_price": 0.15},  # per flyer
    {
        "item_name": "Party streamers",
        "category": "product",
        "unit_price": 0.05,
    },  # per roll
    {
        "item_name": "Decorative adhesive tape (washi tape)",
        "category": "product",
        "unit_price": 0.20,
    },  # per roll
    {
        "item_name": "Paper party bags",
        "category": "product",
        "unit_price": 0.25,
    },  # per bag
    {
        "item_name": "Name tags with lanyards",
        "category": "product",
        "unit_price": 0.75,
    },  # per tag
    {
        "item_name": "Presentation folders",
        "category": "product",
        "unit_price": 0.50,
    },  # per folder
    # Large-format items (priced per unit)
    {
        "item_name": "Large poster paper (24x36 inches)",
        "category": "large_format",
        "unit_price": 1.00,
    },
    {
        "item_name": "Rolls of banner paper (36-inch width)",
        "category": "large_format",
        "unit_price": 2.50,
    },
    # Specialty papers
    {"item_name": "100 lb cover stock", "category": "specialty", "unit_price": 0.50},
    {"item_name": "80 lb text paper", "category": "specialty", "unit_price": 0.40},
    {"item_name": "250 gsm cardstock", "category": "specialty", "unit_price": 0.30},
    {"item_name": "220 gsm poster paper", "category": "specialty", "unit_price": 0.35},
]

DEFAULT_MIN_STOCK_LEVEL = 50

# Given below are some utility functions you can use to implement your multi-agent system


def generate_sample_inventory(
    paper_supplies: list, coverage: float = 0.4, seed: int = 137
) -> pd.DataFrame:
    """
    Generate inventory for exactly a specified percentage of items from the full paper supply list.

    This function randomly selects exactly `coverage` × N items from the `paper_supplies` list,
    and assigns each selected item:
    - a random stock quantity between 200 and 800,
    - a minimum stock level between 50 and 150.

    The random seed ensures reproducibility of selection and stock levels.

    Args:
        paper_supplies (list): A list of dictionaries, each representing a paper item with
                               keys 'item_name', 'category', and 'unit_price'.
        coverage (float, optional): Fraction of items to include in the inventory (default is 0.4, or 40%).
        seed (int, optional): Random seed for reproducibility (default is 137).

    Returns:
        pd.DataFrame: A DataFrame with the selected items and assigned inventory values, including:
                      - item_name
                      - category
                      - unit_price
                      - current_stock
                      - min_stock_level
    """
    # Ensure reproducible random output
    np.random.seed(seed)

    # Calculate number of items to include based on coverage
    num_items = int(len(paper_supplies) * coverage)

    # Randomly select item indices without replacement
    selected_indices = np.random.choice(
        range(len(paper_supplies)), size=num_items, replace=False
    )

    # Extract selected items from paper_supplies list
    selected_items = [paper_supplies[i] for i in selected_indices]

    # Construct inventory records
    inventory = []
    for item in selected_items:
        inventory.append(
            {
                "item_name": item["item_name"],
                "category": item["category"],
                "unit_price": item["unit_price"],
                "current_stock": np.random.randint(200, 800),  # Realistic stock range
                "min_stock_level": np.random.randint(
                    50, 150
                ),  # Reasonable threshold for reordering
            }
        )

    # Return inventory as a pandas DataFrame
    return pd.DataFrame(inventory)


def init_database(db_engine: Engine, seed: int = 137) -> Engine:
    """
    Set up the Munder Difflin database with all required tables and initial records.

    This function performs the following tasks:
    - Creates the 'transactions' table for logging stock orders and sales
    - Loads customer inquiries from 'quote_requests.csv' into a 'quote_requests' table
    - Loads previous quotes from 'quotes.csv' into a 'quotes' table, extracting useful metadata
    - Generates a random subset of paper inventory using `generate_sample_inventory`
    - Inserts initial financial records including available cash and starting stock levels

    Args:
        db_engine (Engine): A SQLAlchemy engine connected to the SQLite database.
        seed (int, optional): A random seed used to control reproducibility of inventory stock levels.
                              Default is 137.

    Returns:
        Engine: The same SQLAlchemy engine, after initializing all necessary tables and records.

    Raises:
        Exception: If an error occurs during setup, the exception is printed and raised.
    """
    try:
        # ----------------------------
        # 1. Create an empty 'transactions' table schema
        # ----------------------------
        transactions_schema = pd.DataFrame(
            {
                "id": [],
                "item_name": [],
                "transaction_type": [],  # 'stock_orders' or 'sales'
                "units": [],  # Quantity involved
                "price": [],  # Total price for the transaction
                "transaction_date": [],  # ISO-formatted date
            }
        )
        transactions_schema.to_sql(
            "transactions", db_engine, if_exists="replace", index=False
        )

        # Set a consistent starting date
        initial_date = datetime(2025, 1, 1).isoformat()

        # ----------------------------
        # 2. Load and initialize 'quote_requests' table
        # ----------------------------
        quote_requests_df = pd.read_csv("quote_requests.csv")
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql(
            "quote_requests", db_engine, if_exists="replace", index=False
        )

        # ----------------------------
        # 3. Load and transform 'quotes' table
        # ----------------------------
        quotes_df = pd.read_csv("quotes.csv")
        quotes_df["request_id"] = range(1, len(quotes_df) + 1)
        quotes_df["order_date"] = initial_date

        # Unpack metadata fields (job_type, order_size, event_type) if present
        if "request_metadata" in quotes_df.columns:
            quotes_df["request_metadata"] = quotes_df["request_metadata"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            quotes_df["job_type"] = quotes_df["request_metadata"].apply(
                lambda x: x.get("job_type", "")
            )
            quotes_df["order_size"] = quotes_df["request_metadata"].apply(
                lambda x: x.get("order_size", "")
            )
            quotes_df["event_type"] = quotes_df["request_metadata"].apply(
                lambda x: x.get("event_type", "")
            )

        # Retain only relevant columns
        quotes_df = quotes_df[
            [
                "request_id",
                "total_amount",
                "quote_explanation",
                "order_date",
                "job_type",
                "order_size",
                "event_type",
            ]
        ]
        quotes_df.to_sql("quotes", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 4. Generate inventory and seed stock
        # ----------------------------
        inventory_df = generate_sample_inventory(paper_supplies, seed=seed)

        # Seed initial transactions
        initial_transactions = []

        # Add a starting cash balance via a dummy sales transaction
        initial_transactions.append(
            {
                "item_name": None,
                "transaction_type": "sales",
                "units": None,
                "price": 50000.0,
                "transaction_date": initial_date,
            }
        )

        # Add one stock order transaction per inventory item
        for _, item in inventory_df.iterrows():
            initial_transactions.append(
                {
                    "item_name": item["item_name"],
                    "transaction_type": "stock_orders",
                    "units": item["current_stock"],
                    "price": item["current_stock"] * item["unit_price"],
                    "transaction_date": initial_date,
                }
            )

        # Commit transactions to database
        pd.DataFrame(initial_transactions).to_sql(
            "transactions", db_engine, if_exists="append", index=False
        )

        # Save the inventory reference table
        inventory_df.to_sql("inventory", db_engine, if_exists="replace", index=False)

        return db_engine

    except Exception as e:
        print(f"Error initializing database: {e}")
        raise


@tool
def create_transaction(
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: str | datetime,
) -> int:
    """
    This function records a transaction of type 'stock_orders' or 'sales' with a specified
    item name, quantity, total price, and transaction date into the 'transactions' table of the database.

    Args:
        item_name (str): The name of the item involved in the transaction.
        transaction_type (str): Either 'stock_orders' or 'sales'.
        quantity (int): Number of units involved in the transaction.
        price (float): Total price of the transaction.
        date (str or datetime): Date of the transaction in ISO 8601 format.

    Returns:
        int: The ID of the newly inserted transaction.

    Raises:
        ValueError: If `transaction_type` is not 'stock_orders' or 'sales'.
        Exception: For other database or execution errors.
    """
    try:
        # Convert datetime to ISO string if necessary
        date_str = date.isoformat() if isinstance(date, datetime) else date

        # Validate transaction type
        if transaction_type not in {"stock_orders", "sales"}:
            raise ValueError("Transaction type must be 'stock_orders' or 'sales'")

        # Prepare transaction record as a single-row DataFrame
        transaction = pd.DataFrame(
            [
                {
                    "item_name": item_name,
                    "transaction_type": transaction_type,
                    "units": quantity,
                    "price": price,
                    "transaction_date": date_str,
                }
            ]
        )

        # Insert the record into the database
        transaction.to_sql("transactions", db_engine, if_exists="append", index=False)

        # Fetch and return the ID of the inserted row
        result = pd.read_sql("SELECT last_insert_rowid() as id", db_engine)
        return int(result.iloc[0]["id"])

    except Exception as e:
        print(f"Error creating transaction: {e}")
        raise


def get_all_inventory(as_of_date: str) -> dict[str, int]:
    """
    Retrieve a snapshot of available inventory as of a specific date.

    This function calculates the net quantity of each item by summing
    all stock orders and subtracting all sales up to and including the given date.

    Only items with positive stock are included in the result.

    Args:
        as_of_date (str): ISO-formatted date string (YYYY-MM-DD) representing the inventory cutoff.

    Returns:
        Dict[str, int]: A dictionary mapping item names to their current stock levels.
    """
    # SQL query to compute stock levels per item as of the given date
    query = """
        SELECT
            item_name,
            SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END) as stock
        FROM transactions
        WHERE item_name IS NOT NULL
        AND transaction_date <= :as_of_date
        GROUP BY item_name
        HAVING stock > 0
    """

    # Execute the query with the date parameter
    result = pd.read_sql(query, db_engine, params={"as_of_date": as_of_date})

    # Convert the result into a dictionary {item_name: stock}
    return dict(zip(result["item_name"], result["stock"]))


@tool
def get_stock_level(item_name: str, as_of_date: str | datetime) -> pd.DataFrame:
    """
    Retrieve the stock level of a specific item as of a given date.

    This function calculates the net stock by summing all 'stock_orders' and
    subtracting all 'sales' transactions for the specified item up to the given date.

    Args:
        item_name (str): The name of the item to look up.
        as_of_date (str or datetime): The cutoff date (inclusive) for calculating stock.

    Returns:
        pd.DataFrame: A single-row DataFrame with columns 'item_name' and 'current_stock'.
    """
    # Convert date to ISO string format if it's a datetime object
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # SQL query to compute net stock level for the item
    stock_query = """
        SELECT
            item_name,
            COALESCE(SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END), 0) AS current_stock
        FROM transactions
        WHERE item_name = :item_name
        AND transaction_date <= :as_of_date
    """

    # Execute query and return result as a DataFrame
    return pd.read_sql(
        stock_query,
        db_engine,
        params={"item_name": item_name, "as_of_date": as_of_date},
    )


@tool
def get_min_stock_level(item_name: str) -> int:
    """
    Retrieve the minimum stock level for a specific item.

    Args:
        item_name (str): The name of the item to look up.

    Returns:
        int: The minimum stock level for the item.
    """
    inventory_df = pd.read_sql(
        "SELECT min_stock_level FROM inventory WHERE item_name = :item_name",
        db_engine,
        params={"item_name": item_name},
    )

    if inventory_df.empty:
        if item_exists(item_name):
            return DEFAULT_MIN_STOCK_LEVEL
        raise ValueError(f"Item not found: {item_name}")

    min_stock = int(inventory_df["min_stock_level"].iloc[0])
    return min_stock


@tool
def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    """
    Estimate the supplier delivery date based on the requested order quantity and a starting date.

    Delivery lead time increases with order size:
        - ≤10 units: same day
        - 11–100 units: 1 day
        - 101–1000 units: 4 days
        - >1000 units: 7 days

    Args:
        input_date_str (str): The starting date in ISO format (YYYY-MM-DD).
        quantity (int): The number of units in the order.

    Returns:
        str: Estimated delivery date in ISO format (YYYY-MM-DD).
    """
    # Debug log (comment out in production if needed)
    """print(
        f"FUNC (get_supplier_delivery_date): Calculating for qty {quantity} from date string '{input_date_str}'"
    )"""

    try:
        input_date_dt = datetime.fromisoformat(input_date_str.split("T")[0])
    except (ValueError, TypeError):
        raise ValueError(
            f"Invalid request date '{input_date_str}'; expected YYYY-MM-DD."
        )

    # Determine delivery delay based on quantity
    if quantity <= 10:
        days = 0
    elif quantity <= 100:
        days = 1
    elif quantity <= 1000:
        days = 4
    else:
        days = 7

    # Add delivery days to the starting date
    delivery_date_dt = input_date_dt + timedelta(days=days)

    # Return formatted delivery date
    return delivery_date_dt.strftime("%Y-%m-%d")


@tool
def get_cash_balance(as_of_date: str | datetime) -> float:
    """
    Calculate the current cash balance as of a specified date.

    The balance is computed by subtracting total stock purchase costs ('stock_orders')
    from total revenue ('sales') recorded in the transactions table up to the given date.

    Args:
        as_of_date (str or datetime): The cutoff date (inclusive) in ISO format or as a datetime object.

    Returns:
        float: Net cash balance as of the given date. Returns 0.0 if no transactions exist or an error occurs.
    """
    try:
        # Convert date to ISO format if it's a datetime object
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.isoformat()

        # Query all transactions on or before the specified date
        transactions = pd.read_sql(
            "SELECT * FROM transactions WHERE transaction_date <= :as_of_date",
            db_engine,
            params={"as_of_date": as_of_date},
        )

        # Compute the difference between sales and stock purchases
        if not transactions.empty:
            total_sales = transactions.loc[
                transactions["transaction_type"] == "sales", "price"
            ].sum()
            total_purchases = transactions.loc[
                transactions["transaction_type"] == "stock_orders", "price"
            ].sum()
            return float(total_sales - total_purchases)

        return 0.0

    except Exception as e:
        print(f"Error getting cash balance: {e}")
        return 0.0


@tool
def generate_financial_report(as_of_date: str | datetime) -> dict:
    """
    Generate a complete financial report for the company as of a specific date.

    This includes:
    - Cash balance
    - Inventory valuation
    - Combined asset total
    - Itemized inventory breakdown
    - Top 5 best-selling products

    Args:
        as_of_date (str or datetime): The date (inclusive) for which to generate the report.

    Returns:
        Dict: A dictionary containing the financial report fields:
            - 'as_of_date': The date of the report
            - 'cash_balance': Total cash available
            - 'inventory_value': Total value of inventory
            - 'total_assets': Combined cash and inventory value
            - 'inventory_summary': List of items with stock and valuation details
            - 'top_selling_products': List of top 5 products by revenue
    """
    # Normalize date input
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # Get current cash balance
    cash = get_cash_balance(as_of_date)

    # Get current inventory snapshot
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inventory_value = 0.0
    inventory_summary = []

    # Compute total inventory value and summary by item
    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(item["item_name"], as_of_date)
        stock = stock_info["current_stock"].iloc[0]
        item_value = stock * item["unit_price"]
        inventory_value += item_value

        inventory_summary.append(
            {
                "item_name": item["item_name"],
                "stock": stock,
                "unit_price": item["unit_price"],
                "value": item_value,
            }
        )

    # Identify top-selling products by revenue
    top_sales_query = """
        SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue
        FROM transactions
        WHERE transaction_type = 'sales' AND transaction_date <= :date
        GROUP BY item_name
        ORDER BY total_revenue DESC
        LIMIT 5
    """
    top_sales = pd.read_sql(top_sales_query, db_engine, params={"date": as_of_date})
    top_selling_products = top_sales.to_dict(orient="records")

    return {
        "as_of_date": as_of_date,
        "cash_balance": cash,
        "inventory_value": inventory_value,
        "total_assets": cash + inventory_value,
        "inventory_summary": inventory_summary,
        "top_selling_products": top_selling_products,
    }


@tool
def search_quote_history(
    search_terms: list[str], limit: int = 5, as_of_date: str = ""
) -> list[dict]:
    """
    Retrieve a list of historical quotes that match any of the provided search terms.

    The function searches both the original customer request (from `quote_requests`) and
    the explanation for the quote (from `quotes`) for each keyword. Results are sorted by
    most recent order date and limited by the `limit` parameter.

    Args:
        search_terms (List[str]): List of terms to match against customer requests and explanations.
        limit (int, optional): Maximum number of quote records to return. Default is 5.
        as_of_date (str, optional): Exclude quotes created after this date.

    Returns:
        List[Dict]: A list of matching quotes, each represented as a dictionary with fields:
            - original_request
            - total_amount
            - quote_explanation
            - job_type
            - order_size
            - event_type
            - order_date
    """
    conditions = []
    params = {}

    # Build SQL WHERE clause using LIKE filters for each search term
    for i, term in enumerate(search_terms):
        param_name = f"term_{i}"
        conditions.append(
            f"(LOWER(qr.response) LIKE :{param_name} OR "
            f"LOWER(q.quote_explanation) LIKE :{param_name})"
        )
        params[param_name] = f"%{term.lower()}%"

    # Combine conditions; fallback to always-true if no terms provided
    where_clause = " AND ".join(conditions) if conditions else "1=1"
    if as_of_date:
        try:
            datetime.fromisoformat(as_of_date.split("T")[0])
        except ValueError as exc:
            raise ValueError(
                f"Invalid request date '{as_of_date}'; expected YYYY-MM-DD."
            ) from exc
        where_clause += " AND q.order_date <= :as_of_date"
        params["as_of_date"] = as_of_date

    # Final SQL query to join quotes with quote_requests
    query = f"""
        SELECT
            qr.response AS original_request,
            q.total_amount,
            q.quote_explanation,
            q.job_type,
            q.order_size,
            q.event_type,
            q.order_date
        FROM quotes q
        JOIN quote_requests qr ON q.request_id = qr.id
        WHERE {where_clause}
        ORDER BY q.order_date DESC
        LIMIT {limit}
    """

    # Execute parameterized query
    with db_engine.connect() as conn:
        result = conn.execute(text(query), params)
        return [dict(row._mapping) for row in result]


########################
########################
########################
# YOUR MULTI AGENT STARTS HERE
########################
########################
########################


# Set up and load your env parameters and instantiate your model.


"""Set up tools for your agents to use, these should be methods that combine the database functions above
 and apply criteria to them to ensure that the flow of the system is correct."""
dotenv.load_dotenv(dotenv_path="../.env")
openai_api_key = os.getenv("UDACITY_OPENAI_API_KEY")
model = OpenAIServerModel(
    model_id="gpt-4o-mini",
    api_base="https://openai.vocareum.com/v1",
    api_key=openai_api_key,
)


# Tools for inventory agent
def item_exists(item_name: str) -> bool:
    """Check if an item exists in the product catalog.

    Args:
        item_name (str): The name of the item to check.

    Returns:
        bool: True if the item is in the catalog, False otherwise.
    """
    return any(item["item_name"] == item_name for item in paper_supplies)


@tool
def get_stock_inventory(
    as_of_date: str, item_name: str, quantity: int, order_date: str
) -> dict[str, Any]:
    """Check if the specified quantity of an item is available in inventory.

    Args:
        as_of_date (str): The date to check inventory for, in YYYY-MM-DD format.
        item_name (str): The name of the item to check.
        quantity (int): The quantity to check for.
        order_date (str): The order date in YYYY-MM-DD format.

    Returns:
        dict[str, Any]: A dictionary containing the item name, whether it exists, the requested quantity, the current stock level, whether there is enough stock, the shortage, and the delivery date.
    """
    if quantity <= 0:
        raise ValueError("Quantity must be greater than zero.")
    if not item_exists(item_name):
        return {"item_name": item_name, "item_exists": False}
    stock_df = get_stock_level(item_name, as_of_date)
    current_stock = int(stock_df["current_stock"].iloc[0])
    shortage = max(0, quantity - current_stock)

    delivery_date = order_date
    if shortage > 0:
        delivery_date = get_supplier_delivery_date(as_of_date, shortage)

    return {
        "item_name": item_name,
        "exists": True,
        "requested_quantity": quantity,
        "current_stock": current_stock,
        "enough_stock": shortage == 0,
        "shortage": shortage,
        "delivery_date": delivery_date,
    }


# Tools for quoting agent


# Tools for ordering agent
@tool
def reorder_item(
    item_name: str, quantity: int, as_of_date: str, order_date: str
) -> dict[str, Any]:
    """Plan separate shortage and safety-stock supplier purchases.

    Args:
        item_name (str): The name of the item to reorder.
        quantity (int): The customer quantity to fulfill.
        as_of_date (str): The order date in YYYY-MM-DD format.
        order_date (str): The order date in YYYY-MM-DD format.

    Returns:
        dict[str, Any]: Separate quantities and delivery dates for each purchase.
    """
    if quantity <= 0:
        raise ValueError("Quantity must be greater than zero.")
    stock = get_stock_inventory(as_of_date, item_name, quantity, order_date=order_date)
    if not stock["exists"]:
        raise ValueError(f"Unknown item: {item_name}")
    shortage_quantity = int(stock["shortage"])
    minimum_stock = get_min_stock_level(item_name)
    remaining_after_sale = max(
        0, int(stock["current_stock"]) + shortage_quantity - quantity
    )
    safety_stock_quantity = max(0, minimum_stock - remaining_after_sale)
    return {
        "item_name": item_name,
        "shortage_quantity": shortage_quantity,
        "shortage_delivery_date": (
            get_supplier_delivery_date(as_of_date, shortage_quantity)
            if shortage_quantity
            else None
        ),
        "safety_stock_quantity": safety_stock_quantity,
        "safety_stock_delivery_date": (
            get_supplier_delivery_date(as_of_date, safety_stock_quantity)
            if safety_stock_quantity
            else None
        ),
        "reordered_quantity": shortage_quantity + safety_stock_quantity,
    }


@tool
def execute_order(
    item_name: str,
    quantity: int,
    order_date: str,
    total_price: float,
) -> list[dict[str, Any]]:
    """Atomically execute shortage, safety-stock, and sales transactions.

    Args:
        item_name: Exact catalog item name.
        quantity: Requested customer quantity.
        order_date: Order date in YYYY-MM-DD format.
        total_price: Total customer sales price.

    Returns:
        A list of the transactions committed for this order.
    """
    if quantity <= 0:
        raise ValueError("Quantity must be greater than zero.")
    catalog_item = next(
        (item for item in paper_supplies if item["item_name"] == item_name), None
    )
    if catalog_item is None:
        raise ValueError(f"Unknown item: {item_name}")

    plan = reorder_item(item_name, quantity, order_date, order_date)
    transactions = []
    with db_engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO inventory (
                    item_name, category, unit_price, current_stock, min_stock_level
                )
                SELECT
                    :item_name, :category, :unit_price, 0, :min_stock_level
                WHERE NOT EXISTS (
                    SELECT 1 FROM inventory WHERE item_name = :item_name
                )
            """),
            {
                "item_name": item_name,
                "category": catalog_item["category"],
                "unit_price": catalog_item["unit_price"],
                "min_stock_level": DEFAULT_MIN_STOCK_LEVEL,
            },
        )
        for transaction_type, units, date in (
            (
                "stock_orders",
                plan["shortage_quantity"],
                plan["shortage_delivery_date"],
            ),
            (
                "stock_orders",
                plan["safety_stock_quantity"],
                plan["safety_stock_delivery_date"],
            ),
            (
                "sales",
                quantity,
                plan["shortage_delivery_date"] or order_date,
            ),
        ):
            if not units:
                continue
            price = (
                float(total_price)
                if transaction_type == "sales"
                else float(units) * float(catalog_item["unit_price"])
            )
            result = conn.execute(
                text("""
                    INSERT INTO transactions (
                        item_name, transaction_type, units, price, transaction_date
                    )
                    VALUES (
                        :item_name, :transaction_type, :units, :price,
                        :transaction_date
                    )
                """),
                {
                    "item_name": item_name,
                    "transaction_type": transaction_type,
                    "units": units,
                    "price": price,
                    "transaction_date": date,
                },
            )
            transactions.append(
                {
                    "transaction_id": int(result.lastrowid),
                    "item_name": item_name,
                    "requested_quantity": int(units),
                    "transaction_type": transaction_type,
                    "status": "completed",
                    "transaction_date": date,
                }
            )
    return transactions


# Set up your agents and create an orchestration agent that will manage them
def parse_agent_value(response: Any) -> Any:
    """Normalize an agent response into a Python JSON-compatible value."""
    if isinstance(response, str):
        try:
            parsed = json.loads(response)
            if isinstance(parsed, str):
                return parse_agent_value(parsed)
            return parsed
        except json.JSONDecodeError as json_error:
            if response.lstrip().startswith("{") and '\\"' in response:
                try:
                    unescaped = json.loads(f'"{response}"')
                    if isinstance(unescaped, str) and unescaped != response:
                        return parse_agent_value(unescaped)
                except json.JSONDecodeError:
                    pass

            repaired = []
            inside_string = False
            escaped = False
            for index, character in enumerate(response):
                if character == "\\" and inside_string and not escaped:
                    escaped = True
                    repaired.append(character)
                    continue
                if character == '"':
                    if not inside_string:
                        inside_string = True
                        repaired.append(character)
                    elif escaped:
                        repaired.append(character)
                    else:
                        remaining = response[index + 1:].lstrip()
                        if remaining.startswith((",", "}", "]", ":")):
                            inside_string = False
                            repaired.append(character)
                        else:
                            repaired.append('\\"')
                    escaped = False
                    continue
                repaired.append(character)
                escaped = False
            repaired_response = "".join(repaired)
            if repaired_response != response:
                try:
                    parsed = json.loads(repaired_response)
                    if isinstance(parsed, str):
                        return parse_agent_value(parsed)
                    return parsed
                except json.JSONDecodeError:
                    pass

            try:
                decoder = json.JSONDecoder()
                candidates = (response.lstrip(), repaired_response.lstrip())
                parsed = None
                for candidate in candidates:
                    try:
                        data, end = decoder.raw_decode(candidate)
                        trailing = candidate[end:].strip()
                    except json.JSONDecodeError:
                        continue
                    if not trailing or set(trailing) <= {"}"}:
                        parsed = data
                        break
                if parsed is not None:
                    return parsed
            except json.JSONDecodeError:
                pass

            python_literal = (
                response.replace("null", "None")
                .replace("true", "True")
                .replace("false", "False")
            )
            try:
                return ast.literal_eval(python_literal)
            except (SyntaxError, ValueError):
                if response.rstrip().endswith("'}"):
                    try:
                        return ast.literal_eval(python_literal[:-2] + '"}')
                    except (SyntaxError, ValueError):
                        pass
                raise json_error
    return response


def parse_agent_response(response: Any) -> dict[str, Any]:
    """Normalize an agent response into a structured dictionary."""
    data = parse_agent_value(response)
    if not isinstance(data, dict):
        raise TypeError("Invalid response format")
    return data


def validate_extracted_request(data: dict[str, Any]) -> dict[str, Any]:
    """Validate catalog mappings without rejecting unrelated valid line items."""
    required_request_fields = {
        "items",
        "request_date",
        "requested_delivery_date",
        "job",
        "event",
    }
    missing_request_fields = required_request_fields - data.keys()
    if missing_request_fields:
        raise ValueError(
            "Missing required request fields: "
            + ", ".join(sorted(missing_request_fields))
        )

    for field in ("request_date", "requested_delivery_date"):
        value = data[field]
        if not isinstance(value, str):
            raise ValueError(f"{field} must be a YYYY-MM-DD string")
        try:
            datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(
                f"Invalid {field} '{value}'; expected YYYY-MM-DD."
            ) from exc
    if datetime.fromisoformat(data["requested_delivery_date"]) < datetime.fromisoformat(
        data["request_date"]
    ):
        raise ValueError("Requested delivery date cannot precede the request date")

    items = data["items"]
    if not isinstance(items, list) or not items:
        raise ValueError("Request must contain at least one item")

    catalog_names = {item["item_name"] for item in paper_supplies}
    validated_items = []
    for item_index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Item {item_index} must be an object")

        original_name = item.get("original_name")
        if not isinstance(original_name, str) or not original_name.strip():
            raise ValueError(f"Item {item_index} is missing original_name")

        validated = dict(item)
        item_name = validated.get("item_name")
        quantity = validated.get("quantity")
        clarification_required = bool(
            validated.get("clarification_required", False)
        )
        reasons = []

        if item_name is None:
            reasons.append("No catalog item was confidently identified")
        elif not isinstance(item_name, str) or item_name not in catalog_names:
            reasons.append(f"'{item_name}' is not an exact catalog item")
            validated["item_name"] = None

        if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity <= 0:
            reasons.append("Quantity could not be converted to a positive catalog unit")
            validated["quantity"] = None

        if reasons:
            clarification_required = True
            existing_message = validated.get("clarification_message")
            validated["clarification_message"] = (
                f"{existing_message} " if isinstance(existing_message, str) else ""
            ) + ". ".join(reasons) + "."

        candidates = validated.get("candidates", [])
        if not isinstance(candidates, list):
            candidates = []
        validated["candidates"] = [
            candidate
            for candidate in candidates
            if isinstance(candidate, str) and candidate in catalog_names
        ]
        validated["original_name"] = original_name.strip()
        validated["clarification_required"] = clarification_required
        validated.setdefault("clarification_message", "")
        requested_attributes = validated.setdefault("requested_attributes", [])
        if not isinstance(requested_attributes, list):
            validated["requested_attributes"] = []
        validated_items.append(validated)

    return {
        **data,
        "items": validated_items,
        "needs_clarification": any(
            item["clarification_required"] for item in validated_items
        ),
    }


def can_fulfill_by_delivery_date(
    inventory: dict[str, Any], requested_delivery_date: str
) -> bool:
    """Return whether inventory can fulfill the item by the requested deadline."""
    if inventory.get("exists") is not True:
        return False
    if inventory.get("enough_stock") is True:
        return True

    delivery_date = inventory.get("delivery_date")
    if not isinstance(delivery_date, str):
        return False
    try:
        return datetime.fromisoformat(delivery_date) <= datetime.fromisoformat(
            requested_delivery_date
        )
    except ValueError as exc:
        raise ValueError("Delivery dates must use ISO format") from exc


def evaluate_workflow_response(response: dict[str, Any]) -> dict[str, Any]:
    """Validate a workflow result and return machine-readable evaluation metrics."""
    valid_statuses = {
        "fulfilled",
        "cannot_deliver_on_time",
        "needs_clarification",
    }
    catalog = {item["item_name"]: item for item in paper_supplies}
    evaluated_items = []
    transaction_ids = []

    datetime.fromisoformat(response["request_date"])
    datetime.fromisoformat(response["requested_delivery_date"])

    for item in response["items"]:
        status = item.get("status")
        if status not in valid_statuses:
            raise AssertionError(f"Unexpected item status: {status!r}")

        item_name = item.get("item_name")
        quote = item.get("quote")
        fulfillment = item.get("fulfillment")
        expected_arithmetic = None
        discount_rate = None

        if status != "needs_clarification":
            if item_name not in catalog:
                raise AssertionError(f"Non-catalog product mapping: {item_name!r}")
        elif item_name is not None:
            raise AssertionError("Clarification item must not have a mapped item")

        if status == "fulfilled":
            if not isinstance(quote, dict):
                raise AssertionError("Fulfilled item is missing its quote")
            if quote.get("item_name") != item_name:
                raise AssertionError("Quote item does not match catalog mapping")

            discount_rate = quote.get("discount")
            if (
                isinstance(discount_rate, bool)
                or not isinstance(discount_rate, (int, float))
                or not 0 <= discount_rate <= 1
            ):
                raise AssertionError("Discount rate must be numeric between 0 and 1")

            unit_price = float(catalog[item_name]["unit_price"])
            quantity = int(quote["requested_quantity"])
            expected_subtotal = round(unit_price * quantity, 2)
            expected_total = round(
                expected_subtotal * (1 - float(discount_rate)), 2
            )
            expected_arithmetic = {
                "unit_price": unit_price,
                "quantity": quantity,
                "expected_subtotal": expected_subtotal,
                "expected_total": expected_total,
            }
            if not abs(float(quote["unit_price"]) - unit_price) < 1e-9:
                raise AssertionError("Quote unit price differs from catalog price")
            if abs(float(quote["subtotal"]) - expected_subtotal) >= 0.01:
                raise AssertionError("Quote subtotal does not match expected amount")
            if abs(float(quote["total_price"]) - expected_total) >= 0.01:
                raise AssertionError("Quote total does not match discount arithmetic")

            if not isinstance(fulfillment, dict):
                raise AssertionError("Fulfilled item is missing fulfillment")
            transactions = fulfillment.get("transactions")
            if not isinstance(transactions, list) or not transactions:
                raise AssertionError("Fulfilled item has no recorded transactions")
            for transaction in transactions:
                transaction_id = transaction.get("transaction_id")
                if not isinstance(transaction_id, int):
                    raise AssertionError("Every transaction must have an integer ID")
                if transaction.get("item_name") != item_name:
                    raise AssertionError("Transaction item differs from product mapping")
                if transaction.get("transaction_type") not in {
                    "sales",
                    "stock_orders",
                }:
                    raise AssertionError("Unexpected transaction type")
                datetime.fromisoformat(transaction["transaction_date"])
                transaction_ids.append(transaction_id)

        evaluated_items.append(
            {
                "original_name": item["original_name"],
                "item_name": item_name,
                "status": status,
                "discount_rate": discount_rate,
                "expected_arithmetic": expected_arithmetic,
                "transaction_ids": (
                    [
                        transaction["transaction_id"]
                        for transaction in fulfillment["transactions"]
                    ]
                    if status == "fulfilled"
                    else []
                ),
            }
        )

    if len(transaction_ids) != len(set(transaction_ids)):
        raise AssertionError("Transaction IDs must be unique within a response")

    return {
        "passed": True,
        "item_count": len(evaluated_items),
        "items": evaluated_items,
        "transaction_count": len(transaction_ids),
    }


def _customer_safe_text(value: Any) -> str:
    """Remove contact details from text that may be shown to a customer."""
    text = str(value or "").strip()
    text = re.sub(
        r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        "[email omitted]",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(?<!\w)(?:\+\d{1,3}[\s.-]?)?"
        r"(?:\(\d{3}\)|\d{3})[\s.-]\d{3}[\s.-]\d{4}(?!\w)",
        "[phone omitted]",
        text,
    )
    technical_replacements = (
        (r"\s*'' is not an exact catalog item\.\s*", " "),
        (
            r"\s*Quantity could not be converted to a positive catalog unit\.\s*",
            " Please provide the quantity in catalog units, such as sheets. ",
        ),
        (
            r"\s*No catalog item was confidently identified\.\s*",
            " Please specify an exact catalog product. ",
        ),
    )
    for pattern, replacement in technical_replacements:
        text = re.sub(pattern, replacement, text)
    return re.sub(r"\s+", " ", text).strip()


def _customer_reason_code(item: dict[str, Any]) -> str:
    status = item.get("status")
    if status == "fulfilled":
        return "ORDER_FULFILLED"
    if status == "cannot_deliver_on_time":
        return "DELIVERY_WINDOW_UNAVAILABLE"
    if status == "needs_clarification":
        message = str(item.get("message", "")).lower()
        if "quantity" in message or "unit" in message or "conversion" in message:
            return "QTY_CLARIFICATION_REQUIRED"
        return "PRODUCT_CLARIFICATION_REQUIRED"
    return "ORDER_REVIEW_REQUIRED"


def _item_customer_description(item: dict[str, Any]) -> str:
    if item.get("item_name"):
        name = str(item["item_name"])
        quantity = item.get("quote", {}).get("requested_quantity")
        if quantity is None:
            quantity = item.get("inventory", {}).get("requested_quantity")
        if quantity is not None:
            return f"{quantity:,} unit(s) of {name}"
        return name
    return _customer_safe_text(item.get("original_name") or "the requested item")


def format_customer_response(response: dict[str, Any]) -> str:
    """Build a customer-safe message without exposing workflow internals."""
    deadline = response.get("requested_delivery_date", "the requested date")
    lines = [
        f"Thank you. We reviewed your request for delivery by {deadline}.",
        "",
        "Order update:",
    ]

    for item in response.get("items", []):
        description = _item_customer_description(item)
        lines.append(f"- {description}")

        if item.get("status") == "fulfilled":
            quote = item["quote"]
            lines.append(f"  Confirmed: ${float(quote['total_price']):.2f}.")
            lines.append("  How we calculated this:")
            lines.append(
                f"  - Catalog price: ${float(quote['unit_price']):.2f} "
                f"x {int(quote['requested_quantity']):,} = "
                f"${float(quote['subtotal']):.2f}."
            )
            lines.append(
                f"  - Bulk discount: {float(quote['discount']) * 100:.0f}%."
            )
            lines.append(
                f"  - Final quoted total: ${float(quote['total_price']):.2f}; "
                "no tax or separate shipping charge is configured."
            )
            transactions = item.get("fulfillment", {}).get("transactions", [])
            transaction_dates = sorted(
                {
                    transaction["transaction_date"]
                    for transaction in transactions
                    if isinstance(transaction, dict)
                    and isinstance(transaction.get("transaction_date"), str)
                }
            )
            if transaction_dates:
                lines.append(
                    f"  - Expected fulfillment date: {transaction_dates[-1]}."
                )
        elif item.get("status") == "cannot_deliver_on_time":
            inventory = item.get("inventory", {})
            available_date = inventory.get("delivery_date") or "a later date"
            lines.append(
                f"  Not confirmed: we can deliver this item on {available_date}, "
                f"not by {deadline}."
            )
            lines.append(
                "  Next step: approve the later date or remove this item "
                "from the request."
            )
        elif item.get("status") == "needs_clarification":
            reason = _customer_safe_text(item.get("message"))
            if not reason:
                reason = "We need one more detail before we can confirm this item."
            lines.append(
                f"  More information needed: {reason}"
            )
            candidates = item.get("candidates") or []
            if candidates:
                candidate_names = [str(candidate) for candidate in candidates[:5]]
                remaining = len(candidates) - len(candidate_names)
                choices = ", ".join(candidate_names)
                if remaining:
                    choices += f" (plus {remaining} related catalog options)"
                lines.append(
                    f"  Suggested choices: {choices}."
                )
            else:
                lines.append(
                    "  Next step: confirm the exact product and quantity in "
                    "catalog units."
                )
        else:
            lines.append(
                "  We could not complete this item. "
                "Please review the request and try again."
            )

    fulfilled = sum(
        item.get("status") == "fulfilled" for item in response.get("items", [])
    )
    total_items = len(response.get("items", []))
    lines.extend(
        [
            "",
            f"Summary: {fulfilled} of {total_items} item(s) confirmed.",
        ]
    )
    if fulfilled < total_items:
        lines.append(
            "The unresolved items listed above do not prevent confirmed items "
            "from being fulfilled."
        )
    return "\n".join(lines)


def customer_status_summary(response: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the minimal structured data needed to audit customer messages."""
    return [
        {
            "item_name": item.get("item_name"),
            "status": item.get("status"),
            "reason_code": _customer_reason_code(item),
            "quantity": (
                item.get("quote", {}).get("requested_quantity")
                or item.get("inventory", {}).get("requested_quantity")
            ),
            "total_price": item.get("quote", {}).get("total_price"),
            "delivery_date": (
                item.get("inventory", {}).get("delivery_date")
                or response.get("requested_delivery_date")
            ),
        }
        for item in response.get("items", [])
    ]


class QuoteResultTool(Tool):
    """Return a schema-constrained discount recommendation."""

    name = "final_answer"
    description = "Return the discount recommendation and pricing explanation."
    output_type = "object"
    inputs = {
        "answer": {
            "type": "object",
            "description": "The quote decision.",
            "properties": {
                "item_name": {"type": "string"},
                "requested_quantity": {"type": "integer"},
                "discount": {
                    "type": "number",
                    "description": "Decimal discount rate from 0 through 1.",
                },
                "pricing_explanation": {"type": "string"},
            },
            "required": [
                "item_name",
                "requested_quantity",
                "discount",
                "pricing_explanation",
            ],
            "additionalProperties": False,
        }
    }

    def forward(self, answer: dict[str, Any]) -> dict[str, Any]:
        return answer


class RequestResultTool(Tool):
    """Return a schema-constrained extracted request from the processor."""

    name = "final_answer"
    description = "Return the extracted customer request."
    output_type = "object"
    inputs = {
        "answer": {
            "type": "object",
            "description": "The complete extracted request.",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "original_name": {"type": "string"},
                            "item_name": {
                                "type": "string",
                                "description": (
                                    "Exact catalog name, or an empty string "
                                    "when clarification is required."
                                ),
                            },
                            "quantity": {
                                "type": "integer",
                                "description": (
                                    "Quantity in catalog units, or zero when "
                                    "the quantity cannot be determined."
                                ),
                            },
                            "requested_attributes": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "candidates": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "clarification_required": {"type": "boolean"},
                            "clarification_message": {"type": "string"},
                        },
                        "required": [
                            "original_name",
                            "item_name",
                            "quantity",
                            "requested_attributes",
                            "candidates",
                            "clarification_required",
                            "clarification_message",
                        ],
                        "additionalProperties": False,
                    },
                },
                "requested_delivery_date": {"type": "string"},
                "job": {"type": "string"},
                "event": {"type": "string"},
                "needs_clarification": {"type": "boolean"},
            },
            "required": [
                "items",
                "needs_clarification",
            ],
            "additionalProperties": False,
        }
    }

    def forward(self, answer: dict[str, Any]) -> dict[str, Any]:
        return answer


class OrderResultTool(Tool):
    """Return the exact transaction array produced by execute_order."""

    name = "final_answer"
    description = "Return the transactions committed for the order."
    output_type = "array"
    inputs = {
        "answer": {
            "type": "array",
            "description": "The exact transaction array returned by execute_order.",
            "items": {
                "type": "object",
                "properties": {
                    "transaction_id": {"type": "integer"},
                    "item_name": {"type": "string"},
                    "requested_quantity": {"type": "integer"},
                    "transaction_type": {
                        "type": "string",
                        "enum": ["stock_orders", "sales"],
                    },
                    "status": {
                        "type": "string",
                        "enum": ["completed"],
                    },
                    "transaction_date": {"type": "string"},
                },
                "required": [
                    "transaction_id",
                    "item_name",
                    "requested_quantity",
                    "transaction_type",
                    "status",
                    "transaction_date",
                ],
                "additionalProperties": False,
            },
        }
    }

    def forward(self, answer: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return answer


class RequestProcessorAgent(ToolCallingAgent):
    """Agent responsible for processing requests."""

    def __init__(self, model):
        super().__init__(
            tools=[RequestResultTool()],
            model=model,
            name="request_processor_agent",
            description="Agent responsible for processing requests and delegating them to the appropriate agents.",
        )


class InventoryAgent(ToolCallingAgent):
    """Agent responsible for managing inventory."""

    def __init__(self, model):
        super().__init__(
            tools=[get_stock_inventory],
            model=model,
            name="inventory_agent",
            description="Agent responsible for managing inventory. Checks stock levels and reorders items when necessary.",
        )


class QuoteAgent(ToolCallingAgent):
    """Agent responsible for handling quote requests."""

    def __init__(self, model):
        super().__init__(
            tools=[
                QuoteResultTool(),
                search_quote_history,
                get_stock_inventory,
                get_cash_balance,
                generate_financial_report,
            ],
            model=model,
            name="quote_agent",
            description="Agent responsible for handling quote requests. Checks stock levels and generates quotes.",
        )


class OrderAgent(ToolCallingAgent):
    """Agent responsible for handling orders."""

    def __init__(self, model):
        super().__init__(
            tools=[
                OrderResultTool(),
                execute_order,
            ],
            model=model,
            name="order_agent",
            description="Agent responsible for creating transactions for orders.",
        )


class OrchestrationAgent(ToolCallingAgent):
    """Agent responsible for orchestrating the interactions between other agents."""

    def __init__(self, model):

        self.request_processor_agent = RequestProcessorAgent(model)
        self.inventory_agent = InventoryAgent(model)
        self.quote_agent = QuoteAgent(model)
        self.order_agent = OrderAgent(model)

        @tool
        def process_request(
            request: str, request_date: str, job: str, event: str
        ) -> dict[str, Any]:
            """Process customer order information to extract details.

            Args:
                request: The customer's order request
                request_date: Trusted request date from the scenario data.
                job: Trusted customer job from the scenario data.
                event: Trusted event from the scenario data.

            Returns:
                Extracted request information, including validated and blocked items.
            """
            response = self.request_processor_agent.run(f"""
                The customer says: "{request}"

                Extract every requested product as a separate line item.
                Select item_name only from this exact catalog:
                {json.dumps(paper_supplies)}

                Rules:
                - Never invent an item_name.
                - Preserve the customer's wording in original_name.
                - Return valid JSON. Escape every quote character inside a string,
                  for example write 24\\" x 36\\" rather than 24" x 36".
                - Keep requested size, color, finish, weight, and other details in
                  requested_attributes.
                - Do not invent attributes that the customer did not state.
                - Convert explicit packages to catalog units when the conversion is
                  stated, such as 10 reams with 500 sheets per ream = 5,000 sheets.
                - Never guess a package conversion.
                - If no single catalog item can be selected confidently, set item_name
                  to "" and quantity to 0 or the explicitly stated quantity when
                  only the catalog mapping is unclear. Set clarification_required
                  to true. Candidate values must
                  be exact names from the supplied catalog. Explain what clarification
                  is needed.
                - A problematic item must not block valid items in the same request.
                - Extract only the requested delivery deadline from the customer
                  text. The request date, customer job, and event are already
                  supplied as trusted context and must not be extracted from the
                  customer wording.

                Return only valid JSON:
                {{
                  "items": [
                    {{
                      "original_name": "customer's product wording",
                      "item_name": "exact catalog name or empty string",
                      "quantity": positive_integer_in_catalog_units_or_0,
                      "requested_attributes": ["A4", "white"],
                      "candidates": ["possible exact catalog names"],
                      "clarification_required": false,
                      "clarification_message": ""
                    }}
                  ],
                  "requested_delivery_date": "YYYY-MM-DD",
                  "needs_clarification": false
                }}
                """)
            data = parse_agent_response(response)
            data["request_date"] = request_date
            data["job"] = job
            data["event"] = event
            return validate_extracted_request(data)

        @tool
        def check_inventory(
            item_name: str,
            quantity: int,
            order_date: str,
        ) -> dict[str, Any]:
            """Check the inventory for the requested items.

            Args:
                item_name: Exact catalog item name.
                quantity: Number of units requested.
                order_date: Request date in YYYY-MM-DD format.

            Returns:
                Inventory status for the requested items.
            """
            request_information = {
                "item_name": item_name,
                "quantity": quantity,
                "order_date": order_date,
            }
            response = self.inventory_agent.run(f"""
                The customer wants: "{request_information}"
                Use the tool get_stock_inventory and get the following information:
                    Check if the requested items are available in the inventory.
                    if it is available check the if the current stock is enough to fulfill the requested quantity.
                    if the current stock is not enough to fulfill the requested quantity, calculate the shortage then determine the estimated delivery date for the shortage.
                    if there is no shortage, the delivery date is the same as the request date.

                Return only valid JSON with item_name, exists, requested_quantity, current_stock, enough_stock, shortage, and delivery_date.
                Use double quotes, JSON booleans, and null where appropriate.
                """)
            data = parse_agent_response(response)
            if (
                "item_name" not in data
                or "exists" not in data
                or "requested_quantity" not in data
                or "current_stock" not in data
                or "enough_stock" not in data
                or "shortage" not in data
                or "delivery_date" not in data
            ):
                raise ValueError("Missing required fields in response")
            return data

        @tool
        def get_pricing(
            item_name: str,
            quantity: int,
            order_date: str,
            job: str,
            event: str,
        ) -> dict[str, Any]:
            """Get the pricing for the requested items.

            Args:
                item_name: Exact catalog item name.
                quantity: Number of units requested.
                order_date: Request date in YYYY-MM-DD format.
                job: Customer's job or role.
                event: Customer event or occasion.

            Returns:
                Pricing information for the requested items.
            """
            catalog_item = next(
                (item for item in paper_supplies if item["item_name"] == item_name),
                None,
            )
            if catalog_item is None:
                raise ValueError(f"Unknown item: {item_name}")
            request_information = {
                "item_name": item_name,
                "quantity": quantity,
                "order_date": order_date,
                "job": job,
                "event": event,
                "unit_price": catalog_item["unit_price"],
            }
            response = self.quote_agent.run(f"""
                The customer wants: "{request_information}"

                Recommend a discount as a decimal between 0 and 1.
                When searching quote history, include only quotes on or before
                the request's order_date.
                Explain the exact pricing logic in pricing_explanation, including:
                - catalog unit price and base subtotal
                - selected bulk discount rate and why
                - relevant quote history and financial context considered
                - that Python will calculate the final total
                Return only valid JSON with item_name, requested_quantity,
                discount, and pricing_explanation.
                Use double quotes around all keys and string values.
                """)
            data = parse_agent_response(response)
            if (
                "item_name" not in data
                or "requested_quantity" not in data
                or "discount" not in data
                or "pricing_explanation" not in data
            ):
                raise ValueError("Missing required fields in response")
            discount = data["discount"]
            if (
                isinstance(discount, bool)
                or not isinstance(discount, (int, float))
                or not 0 <= discount <= 1
            ):
                raise ValueError("Discount must be a numeric rate between 0 and 1")
            if (
                not isinstance(data["pricing_explanation"], str)
                or not data["pricing_explanation"].strip()
            ):
                raise ValueError("Pricing explanation must be a non-empty string")
            data["unit_price"] = catalog_item["unit_price"]
            data["subtotal"] = round(catalog_item["unit_price"] * quantity, 2)
            data["total_price"] = round(
                data["subtotal"] * (1 - float(discount)), 2
            )
            return data

        @tool
        def finalize_order(
            item_name: str,
            quantity: int,
            order_date: str,
            total_price: float,
        ) -> dict[str, Any]:
            """Fulfill the order for the requested items.

            Args:
                item_name: Exact catalog item name.
                quantity: Number of units to fulfill.
                order_date: Order date in YYYY-MM-DD format.
                total_price: Quoted total price for the order.

            Returns:
                Confirmation of the order fulfillment.
            """
            request_information = {
                "item_name": item_name,
                "quantity": quantity,
                "order_date": order_date,
                "total_price": total_price,
            }
            response = self.order_agent.run(f"""
                The customer wants: "{request_information}"
                Call execute_order exactly once with these exact arguments.
                This tool atomically records any shortage purchase, safety-stock
                purchase, and the customer sale.
                Return exactly the transaction array produced by execute_order.
                Do not call any other transaction tool or repeat the call.
                """)
            data = parse_agent_value(response)
            if isinstance(data, dict):
                transactions = [data]
                original_shape_is_dict = True
            else:
                transactions = data
                original_shape_is_dict = False
            if not isinstance(transactions, list) or not transactions:
                raise TypeError("Invalid fulfillment response format")

            required_fields = {
                "item_name",
                "requested_quantity",
                "transaction_type",
                "status",
                "transaction_date",
            }
            for transaction in transactions:
                if not isinstance(transaction, dict):
                    raise TypeError("Each fulfillment transaction must be an object")
                if required_fields - transaction.keys():
                    raise ValueError("Missing required fulfillment fields")

            if original_shape_is_dict:
                return data

            return {
                "item_name": item_name,
                "requested_quantity": quantity,
                "transaction_type": (
                    transactions[0]["transaction_type"]
                    if len(transactions) == 1
                    else "multiple"
                ),
                "status": (
                    "fulfilled"
                    if all(transaction["status"] for transaction in transactions)
                    else "failed"
                ),
                "transactions": transactions,
            }

        super().__init__(
            tools=[
                process_request,
                check_inventory,
                get_pricing,
                finalize_order,
            ],
            model=model,
            name="orchestration_agent",
            description="Agent responsible for orchestrating the interactions between other agents.",
        )

    def handle_request(
        self, request: str, request_date: str, job: str, event: str,
        request_id: str | None = None,
    ):
        request_id = request_id or f"request-{request_date}"

        def report_progress(
            event_type: str,
            stage: str,
            description: str,
            item_id: str | None = None,
            details: dict[str, Any] | None = None,
        ) -> None:
            emitter.emit(WorkflowEvent(
                event_type=event_type,
                stage=stage,
                timestamp=datetime.now(),
                description=description,
                request_id=request_id,
                item_id=item_id,
                details=details,
            ))

        report_progress(
            "extraction", "started", "Extracting requested products and dates."
        )
        processed = self.tools["process_request"](
            request=request,
            request_date=request_date,
            job=job,
            event=event,
        )
        report_progress(
            "extraction",
            "completed",
            f"Extracted {len(processed['items'])} requested item(s).",
            details={
                "item_count": len(processed["items"]),
                "needs_clarification": processed["needs_clarification"],
                "delivery_date": processed["requested_delivery_date"],
            },
        )
        item_results = []

        for item in processed["items"]:
            if item["clarification_required"]:
                item_results.append(
                    {
                        "original_name": item["original_name"],
                        "item_name": None,
                        "status": "needs_clarification",
                        "message": item["clarification_message"],
                        "candidates": item["candidates"],
                    }
                )
                continue

            report_progress(
                "inventory",
                "started",
                f"Checking inventory for {item['item_name']}.",
                item["item_name"],
            )
            inventory = self.tools["check_inventory"](
                item_name=item["item_name"],
                quantity=item["quantity"],
                order_date=processed["request_date"],
            )
            inventory["order_will_be_fulfilled_on_time"] = (
                can_fulfill_by_delivery_date(
                    inventory, processed["requested_delivery_date"]
                )
            )
            report_progress(
                "inventory",
                "completed",
                (
                    f"{item['item_name']} inventory verified; "
                    f"{inventory['current_stock']} currently available."
                ),
                item["item_name"],
                details=inventory,
            )
            if not inventory["order_will_be_fulfilled_on_time"]:
                item_results.append(
                    {
                        "original_name": item["original_name"],
                        "item_name": item["item_name"],
                        "inventory": inventory,
                        "status": "cannot_deliver_on_time",
                        "message": (
                            "The item cannot be delivered by the requested "
                            "delivery date. "
                            f"Estimated delivery: {inventory['delivery_date']}."
                        ),
                        "required_next_action": (
                            "Ask the customer to provide a later delivery date "
                            "or remove this item from the request."
                        ),
                    }
                )
                continue

            report_progress(
                "pricing",
                "started",
                f"Calculating a quote for {item['item_name']}.",
                item["item_name"],
            )
            pricing = self.tools["get_pricing"](
                item_name=item["item_name"],
                quantity=item["quantity"],
                order_date=processed["request_date"],
                job=processed["job"],
                event=processed["event"],
            )
            report_progress(
                "pricing",
                "completed",
                f"Calculated ${pricing['total_price']:.2f} for {item['item_name']}.",
                item["item_name"],
                details=pricing,
            )

            result = {
                "original_name": item["original_name"],
                "item_name": item["item_name"],
                "inventory": inventory,
                "quote": pricing,
                "pricing_explanation": pricing["pricing_explanation"],
            }
            report_progress(
                "fulfillment",
                "started",
                f"Executing the order for {item['item_name']}.",
                item["item_name"],
            )
            result["fulfillment"] = self.tools["finalize_order"](
                item_name=item["item_name"],
                quantity=item["quantity"],
                order_date=processed["request_date"],
                total_price=pricing["total_price"],
            )
            result["status"] = "fulfilled"
            item_results.append(result)
            report_progress(
                "fulfillment",
                "completed",
                f"Fulfilled {item['quantity']} unit(s) of {item['item_name']}.",
                item["item_name"],
                details=result["fulfillment"],
            )

        return {
            "request_date": processed["request_date"],
            "requested_delivery_date": processed["requested_delivery_date"],
            "job": processed["job"],
            "event": processed["event"],
            "needs_clarification": processed["needs_clarification"],
            "items": item_results,
        }


# Run your test scenarios by writing them here. Make sure to keep track of them.


def run_test_scenarios():

    print("Initializing Database...")
    init_database(db_engine=db_engine)
    try:
        quote_requests_sample = pd.read_csv("quote_requests_sample.csv")
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"], format="%m/%d/%y", errors="coerce"
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date")
    except Exception as e:
        print(f"FATAL: Error loading test data: {e}")
        return

    # Get initial state
    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]

    ############
    ############
    ############
    # INITIALIZE YOUR MULTI AGENT SYSTEM HERE
    ############
    ############
    ############
    orchestration_agent = OrchestrationAgent(model)
    visualizer = TerminalVisualizer(max_history=8)
    emitter.subscribe(visualizer.on_event)

    results = []
    debug_results = []
    for idx, row in quote_requests_sample.iterrows():
        request_date = row["request_date"].strftime("%Y-%m-%d")

        print(f"\n=== Request {idx + 1} ===")
        print(f"Context: {row['job']} organizing {row['event']}")
        print(f"Request Date: {request_date}")
        print(f"Cash Balance: ${current_cash:.2f}")
        print(f"Inventory Value: ${current_inventory:.2f}")

        ############
        ############
        ############
        # USE YOUR MULTI AGENT SYSTEM TO HANDLE THE REQUEST
        ############
        ############
        ############

        try:
            response = orchestration_agent.handle_request(
                request=row["request"],
                request_date=request_date,
                job=row["job"],
                event=row["event"],
                request_id=str(idx + 1),
            )
            response["evaluation"] = evaluate_workflow_response(response)
            customer_response = format_customer_response(response)
            status_summary = customer_status_summary(response)
        except Exception as exc:
            print(
                f"Request {idx + 1} failed internally "
                f"({type(exc).__name__}): {exc}"
            )
            customer_response = (
                "We could not finish processing this request. "
                "Please try again with a smaller quantity or confirm the exact "
                "product and delivery date."
            )
            status_summary = [
                {
                    "status": "error",
                    "reason_code": "REQUEST_PROCESSING_ERROR",
                }
            ]
            response = {
                "evaluation": {
                    "passed": False,
                    "reason_code": "REQUEST_PROCESSING_ERROR",
                }
            }

        # Update state
        report = generate_financial_report(request_date)
        current_cash = report["cash_balance"]
        current_inventory = report["inventory_value"]

        print(f"\nCustomer response:\n{customer_response}")
        print(f"Updated Cash: ${current_cash:.2f}")
        print(f"Updated Inventory: ${current_inventory:.2f}")

        results.append(
            {
                "request_id": idx + 1,
                "request_date": request_date,
                "response": customer_response,
            }
        )
        debug_results.append(
            {
                "request_id": int(idx + 1),
                "request_date": request_date,
                "status_summary": status_summary,
                "evaluation": response["evaluation"],
            }
        )

        time.sleep(1)

    # Final report
    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    print("\n===== FINAL FINANCIAL REPORT =====")
    print(f"Final Cash: ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory: ${final_report['inventory_value']:.2f}")

    # Save results
    pd.DataFrame(results).to_csv("test_results.csv", index=False)
    with open("test_results_debug.json", "w", encoding="utf-8") as debug_file:
        json.dump(debug_results, debug_file, indent=2, ensure_ascii=False)
    emitter.unsubscribe(visualizer.on_event)
    return results


if __name__ == "__main__":
    results = run_test_scenarios()
