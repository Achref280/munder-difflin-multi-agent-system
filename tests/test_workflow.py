import sys
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
from sqlalchemy import create_engine, text

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import project_starter as project


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.original_engine = project.db_engine
        self.engine = create_engine("sqlite:///:memory:")
        project.db_engine = self.engine

        with self.engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    item_name TEXT,
                    transaction_type TEXT,
                    units INTEGER,
                    price REAL,
                    transaction_date TEXT
                )
            """))
            conn.execute(text("""
                CREATE TABLE inventory (
                    item_name TEXT,
                    category TEXT,
                    unit_price REAL,
                    current_stock INTEGER,
                    min_stock_level INTEGER
                )
            """))
            conn.execute(text("""
                CREATE TABLE quote_requests (id INTEGER, response TEXT)
            """))
            conn.execute(text("""
                CREATE TABLE quotes (
                    request_id INTEGER,
                    total_amount REAL,
                    quote_explanation TEXT,
                    job_type TEXT,
                    order_size TEXT,
                    event_type TEXT,
                    order_date TEXT
                )
            """))
            conn.execute(text("""
                INSERT INTO inventory
                    (item_name, category, unit_price, current_stock, min_stock_level)
                VALUES ('A4 paper', 'paper', 0.05, 5, 2)
            """))
            conn.execute(text("""
                INSERT INTO transactions
                    (item_name, transaction_type, units, price, transaction_date)
                VALUES
                    ('A4 paper', 'stock_orders', 5, 0.25, '2025-01-01'),
                    ('A4 paper', 'sales', 2, 0.20, '2025-01-01'),
                    ('A4 paper', 'sales', 1, 0.10, '2025-01-02')
            """))
            conn.execute(text("""
                INSERT INTO quote_requests (id, response)
                VALUES (1, 'A4 paper for an office')
            """))
            conn.execute(text("""
                INSERT INTO quotes (
                    request_id, total_amount, quote_explanation, job_type,
                    order_size, event_type, order_date
                )
                VALUES (
                    1, 10.0, 'A4 paper quote', 'office manager',
                    'small', 'meeting', '2025-01-01'
                )
            """))

    def tearDown(self):
        project.db_engine = self.original_engine
        self.engine.dispose()

    def test_inventory_and_cash_as_of_date(self):
        stock = project.get_stock_level("A4 paper", "2025-01-01")
        self.assertEqual(int(stock["current_stock"].iloc[0]), 3)
        self.assertEqual(project.get_all_inventory("2025-01-01"), {"A4 paper": 3})
        self.assertAlmostEqual(project.get_cash_balance("2025-01-01"), -0.05)
        self.assertEqual(
            int(project.get_stock_level("A4 paper", "2025-01-02")
                ["current_stock"].iloc[0]),
            2,
        )

    def test_catalog_and_inventory_tool_for_known_and_unknown_items(self):
        self.assertTrue(project.item_exists("A4 paper"))
        self.assertTrue(project.item_exists("Paper cups"))
        self.assertFalse(project.item_exists("Not a catalog item"))

        available = project.get_stock_inventory("2025-01-01", "A4 paper", 2)
        self.assertTrue(available["exists"])
        self.assertEqual(available["current_stock"], 3)
        self.assertTrue(available["enough_stock"])
        self.assertEqual(available["shortage"], 0)

        unknown = project.get_stock_inventory(
            "2025-01-01", "Not a catalog item", 2
        )
        self.assertEqual(
            unknown,
            {"item_name": "Not a catalog item", "item_exists": False},
        )

    def test_inventory_agent_tool_reports_shortage_and_delivery_date(self):
        result = project.get_stock_inventory("2025-01-01", "A4 paper", 10)
        self.assertTrue(result["exists"])
        self.assertEqual(result["requested_quantity"], 10)
        self.assertEqual(result["current_stock"], 3)
        self.assertEqual(result["shortage"], 7)
        self.assertEqual(result["delivery_date"], "2025-01-01")

    def test_leaf_agents_register_their_tools(self):
        expected_tools = {
            project.InventoryAgent: {
                "get_stock_inventory",
                "get_min_stock_level",
                "create_transaction",
            },
            project.QuoteAgent: {
                "search_quote_history",
                "get_stock_inventory",
                "get_cash_balance",
                "generate_financial_report",
            },
            project.OrderAgent: {"reorder_item", "create_transaction"},
        }
        for agent_type, expected in expected_tools.items():
            with self.subTest(agent=agent_type.__name__):
                agent = agent_type(Mock())
                self.assertEqual(set(agent.tools) - {"final_answer"}, expected)

    def test_orchestrator_registers_its_workflow_tools(self):
        agent = project.OrchestrationAgent(Mock())
        self.assertEqual(
            set(agent.tools) - {"final_answer"},
            {"process_request", "check_inventory", "get_pricing", "finalize_order", "reorder_stock"},
        )

    def test_minimum_stock_lookup(self):
        self.assertEqual(project.get_min_stock_level("A4 paper"), 2)
        with self.assertRaisesRegex(ValueError, "Item not found"):
            project.get_min_stock_level("Paper cups")

    def test_reorder_rejects_nonpositive_quantity(self):
        with self.assertRaisesRegex(ValueError, "Quantity must be greater than zero"):
            project.reorder_item("A4 paper", 0, "2025-01-01")

    def test_order_agent_tool_reorders_shortage_and_restores_minimum_stock(self):
        stock = {
            "exists": True,
            "current_stock": 5,
            "shortage": 15,
            "min_stock_level": 2,
        }
        with patch.object(project, "get_stock_inventory", return_value=stock):
            result = project.reorder_item("A4 paper", 20, "2025-01-01")
        self.assertEqual(result["reordered_quantity"], 17)
        self.assertEqual(result["delivery_date"], "2025-01-02")

    def test_supplier_delivery_lead_time_boundaries(self):
        cases = {
            10: "2025-01-01",
            11: "2025-01-02",
            100: "2025-01-02",
            101: "2025-01-05",
            1000: "2025-01-05",
            1001: "2025-01-08",
        }
        for quantity, expected_date in cases.items():
            with self.subTest(quantity=quantity):
                self.assertEqual(
                    project.get_supplier_delivery_date("2025-01-01", quantity),
                    expected_date,
                )

    def test_create_transaction_persists_valid_transaction_and_rejects_bad_type(self):
        transaction_id = project.create_transaction(
            "A4 paper", "sales", 1, 0.05, datetime(2025, 1, 3)
        )
        self.assertIsInstance(transaction_id, int)

        with self.engine.connect() as conn:
            row = conn.execute(
                text("""
                    SELECT item_name, transaction_type, units, price, transaction_date
                    FROM transactions WHERE id = :id
                """),
                {"id": transaction_id},
            ).one()
        self.assertEqual(
            tuple(row),
            ("A4 paper", "sales", 1, 0.05, "2025-01-03T00:00:00"),
        )

        with self.assertRaisesRegex(ValueError, "Transaction type"):
            project.create_transaction("A4 paper", "refund", 1, 0.05, "2025-01-03")

    def test_financial_report_aggregates_inventory_and_sales(self):
        report = project.generate_financial_report("2025-01-01")
        self.assertAlmostEqual(report["cash_balance"], -0.05)
        self.assertAlmostEqual(report["inventory_value"], 0.15)
        self.assertAlmostEqual(report["total_assets"], 0.10)
        self.assertEqual(
            report["top_selling_products"],
            [{"item_name": "A4 paper", "total_units": 2, "total_revenue": 0.2}],
        )

    def test_quote_history_search_and_limit(self):
        history = project.search_quote_history(["A4 paper"], limit=1)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["original_request"], "A4 paper for an office")
        self.assertEqual(history[0]["total_amount"], 10.0)

    def test_sample_inventory_is_reproducible(self):
        first = project.generate_sample_inventory(project.paper_supplies, seed=7)
        second = project.generate_sample_inventory(project.paper_supplies, seed=7)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(len(first), int(len(project.paper_supplies) * 0.4))
        self.assertTrue(first["current_stock"].between(200, 799).all())
        self.assertTrue(first["min_stock_level"].between(50, 149).all())

    def test_database_initialization_loads_quote_metadata(self):
        requests = pd.DataFrame([{"response": "A4 paper for an office"}])
        quotes = pd.DataFrame([{
            "total_amount": 10.0,
            "quote_explanation": "A4 paper quote",
            "request_metadata": (
                "{'job_type': 'office manager', 'order_size': 'small', "
                "'event_type': 'meeting'}"
            ),
        }])
        engine = create_engine("sqlite:///:memory:")
        try:
            with patch.object(
                project.pd, "read_csv", side_effect=[requests, quotes]
            ):
                self.assertIs(project.init_database(engine, seed=7), engine)

            loaded_quote = pd.read_sql("SELECT * FROM quotes", engine).iloc[0]
            self.assertEqual(loaded_quote["request_id"], 1)
            self.assertEqual(loaded_quote["job_type"], "office manager")
            self.assertEqual(loaded_quote["order_size"], "small")
            self.assertEqual(loaded_quote["event_type"], "meeting")

            transaction_count = pd.read_sql(
                "SELECT COUNT(*) AS count FROM transactions", engine
            ).iloc[0]["count"]
            inventory_count = pd.read_sql(
                "SELECT COUNT(*) AS count FROM inventory", engine
            ).iloc[0]["count"]
            self.assertEqual(transaction_count, inventory_count + 1)
        finally:
            engine.dispose()


if __name__ == "__main__":
    unittest.main()
