import json
import os
import sys
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import project_starter as project


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.original_engine = project.db_engine
        self.engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        project.db_engine = self.engine

        with self.engine.begin() as conn:
            conn.execute(
                text("""
                CREATE TABLE transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    item_name TEXT,
                    transaction_type TEXT,
                    units INTEGER,
                    price REAL,
                    transaction_date TEXT
                )
            """)
            )
            conn.execute(
                text("""
                CREATE TABLE inventory (
                    item_name TEXT,
                    category TEXT,
                    unit_price REAL,
                    current_stock INTEGER,
                    min_stock_level INTEGER
                )
            """)
            )
            conn.execute(
                text("""
                CREATE TABLE quote_requests (id INTEGER, response TEXT)
            """)
            )
            conn.execute(
                text("""
                CREATE TABLE quotes (
                    request_id INTEGER,
                    total_amount REAL,
                    quote_explanation TEXT,
                    job_type TEXT,
                    order_size TEXT,
                    event_type TEXT,
                    order_date TEXT
                )
            """)
            )
            conn.execute(
                text("""
                INSERT INTO inventory
                    (item_name, category, unit_price, current_stock, min_stock_level)
                VALUES ('A4 paper', 'paper', 0.05, 5, 2)
            """)
            )
            conn.execute(
                text("""
                INSERT INTO transactions
                    (item_name, transaction_type, units, price, transaction_date)
                VALUES
                    ('A4 paper', 'stock_orders', 5, 0.25, '2025-01-01'),
                    ('A4 paper', 'sales', 2, 0.20, '2025-01-01'),
                    ('A4 paper', 'sales', 1, 0.10, '2025-01-02')
            """)
            )
            conn.execute(
                text("""
                INSERT INTO quote_requests (id, response)
                VALUES (1, 'A4 paper for an office')
            """)
            )
            conn.execute(
                text("""
                INSERT INTO quotes (
                    request_id, total_amount, quote_explanation, job_type,
                    order_size, event_type, order_date
                )
                VALUES (
                    1, 10.0, 'A4 paper quote', 'office manager',
                    'small', 'meeting', '2025-01-01'
                )
            """)
            )

    def tearDown(self):
        project.db_engine = self.original_engine
        self.engine.dispose()

    def test_inventory_and_cash_as_of_date(self):
        stock = project.get_stock_level("A4 paper", "2025-01-01")
        self.assertEqual(int(stock["current_stock"].iloc[0]), 3)
        self.assertEqual(project.get_all_inventory("2025-01-01"), {"A4 paper": 3})
        self.assertAlmostEqual(project.get_cash_balance("2025-01-01"), -0.05)
        self.assertEqual(
            int(
                project.get_stock_level("A4 paper", "2025-01-02")["current_stock"].iloc[
                    0
                ]
            ),
            2,
        )

    def test_catalog_and_inventory_tool_for_known_and_unknown_items(self):
        self.assertTrue(project.item_exists("A4 paper"))
        self.assertTrue(project.item_exists("Paper cups"))
        self.assertFalse(project.item_exists("Not a catalog item"))

        available = project.get_stock_inventory(
            "2025-01-01", "A4 paper", 2, "2025-01-01"
        )
        self.assertTrue(available["exists"])
        self.assertEqual(available["current_stock"], 3)
        self.assertTrue(available["enough_stock"])
        self.assertEqual(available["shortage"], 0)

        unknown = project.get_stock_inventory(        "2025-01-01", "Not a catalog item", 2, "2025-01-01")
        self.assertEqual(
            unknown,
            {"item_name": "Not a catalog item", "item_exists": False},
        )

    def test_inventory_agent_tool_reports_shortage_and_delivery_date(self):
        result = project.get_stock_inventory(
            "2025-01-01", "A4 paper", 10, "2025-01-01"
        )
        self.assertTrue(result["exists"])
        self.assertEqual(result["requested_quantity"], 10)
        self.assertEqual(result["current_stock"], 3)
        self.assertEqual(result["shortage"], 7)
        self.assertEqual(result["delivery_date"], "2025-01-01")

    def test_leaf_agents_register_their_tools(self):
        expected_tools = {
            project.InventoryAgent: {
                "get_stock_inventory",
            },
            project.QuoteAgent: {
                "search_quote_history",
                "get_stock_inventory",
                "get_cash_balance",
                "generate_financial_report",
            },
            project.OrderAgent: {
                "execute_order",
            },
        }
        for agent_type, expected in expected_tools.items():
            with self.subTest(agent=agent_type.__name__):
                agent = agent_type(Mock())
                self.assertEqual(set(agent.tools) - {"final_answer"}, expected)

    def test_request_processor_uses_structured_final_answer_schema(self):
        agent = project.RequestProcessorAgent(Mock())
        final_answer = agent.tools["final_answer"]

        self.assertIsInstance(final_answer, project.RequestResultTool)
        answer_schema = final_answer.inputs["answer"]
        self.assertEqual(answer_schema["type"], "object")
        self.assertIn("items", answer_schema["properties"])
        self.assertFalse(answer_schema["additionalProperties"])
        item_schema = answer_schema["properties"]["items"]["items"]
        self.assertIn("clarification_required", item_schema["required"])
        self.assertEqual(
            item_schema["properties"]["item_name"]["type"], "string"
        )

    def test_quote_agent_uses_structured_final_answer_schema(self):
        agent = project.QuoteAgent(Mock())
        final_answer = agent.tools["final_answer"]
        self.assertIsInstance(final_answer, project.QuoteResultTool)
        answer_schema = final_answer.inputs["answer"]
        self.assertEqual(answer_schema["properties"]["discount"]["type"], "number")
        self.assertIn("pricing_explanation", answer_schema["required"])

    def test_order_agent_uses_structured_final_answer_schema(self):
        agent = project.OrderAgent(Mock())
        final_answer = agent.tools["final_answer"]
        self.assertIsInstance(final_answer, project.OrderResultTool)
        answer_schema = final_answer.inputs["answer"]
        self.assertEqual(answer_schema["type"], "array")
        transaction_schema = answer_schema["items"]
        self.assertEqual(
            transaction_schema["properties"]["transaction_type"]["enum"],
            ["stock_orders", "sales"],
        )
        self.assertIn("transaction_id", transaction_schema["required"])

    def test_orchestrator_registers_its_workflow_tools(self):
        agent = project.OrchestrationAgent(Mock())
        self.assertEqual(
            set(agent.tools) - {"final_answer"},
            {
                "process_request",
                "check_inventory",
                "get_pricing",
                "finalize_order",
            },
        )

    def test_terminal_listener_does_not_change_workflow_result(self):
        agent = project.OrchestrationAgent(Mock())
        processed = {
            "items": [
                {
                    "original_name": "10 A4 sheets",
                    "item_name": "A4 paper",
                    "quantity": 10,
                    "requested_attributes": ["A4"],
                    "candidates": [],
                    "clarification_required": False,
                    "clarification_message": "",
                }
            ],
            "request_date": "2025-01-01",
            "requested_delivery_date": "2025-01-05",
            "job": "manager",
            "event": "meeting",
            "needs_clarification": False,
        }
        inventory = {
            "item_name": "A4 paper",
            "exists": True,
            "requested_quantity": 10,
            "current_stock": 10,
            "enough_stock": True,
            "shortage": 0,
            "delivery_date": "2025-01-01",
            "order_will_be_fulfilled_on_time": True,
        }
        pricing = {
            "item_name": "A4 paper",
            "requested_quantity": 10,
            "discount": 0,
            "total_price": 0.5,
            "pricing_explanation": "Base price.",
        }
        fulfillment = {
            "item_name": "A4 paper",
            "requested_quantity": 10,
            "transaction_type": "sales",
            "status": "fulfilled",
            "transactions": [],
        }
        project.emitter.clear()

        def broken_listener(_event):
            raise RuntimeError("Terminal display failed")

        project.emitter.subscribe(broken_listener)
        try:
            with (
                patch.object(
                    agent.tools["process_request"], "forward",
                    return_value=processed,
                ),
                patch.object(
                    agent.tools["check_inventory"], "forward",
                    return_value=inventory,
                ),
                patch.object(
                    agent.tools["get_pricing"], "forward",
                    return_value=pricing,
                ),
                patch.object(
                    agent.tools["finalize_order"], "forward",
                    return_value=fulfillment,
                ),
            ):
                result = agent.handle_request(
                    "A4 paper request",
                    "2025-01-01",
                    "manager",
                    "meeting",
                    request_id="REQ-001",
                )
        finally:
            project.emitter.unsubscribe(broken_listener)

        self.assertEqual(result["items"][0]["fulfillment"], fulfillment)
        self.assertEqual(result["items"][0]["status"], "fulfilled")
        event_types = [
            event.event_type
            for event in project.emitter.get_events(request_id="REQ-001")
        ]
        self.assertEqual(
            event_types,
            [
                "extraction",
                "extraction",
                "inventory",
                "inventory",
                "pricing",
                "pricing",
                "fulfillment",
                "fulfillment",
            ],
        )

    def test_customer_response_is_readable_explainable_and_safe(self):
        response = {
            "request_date": "2025-01-01",
            "requested_delivery_date": "2025-01-10",
            "job": "private customer role",
            "event": "private occasion",
            "items": [
                {
                    "original_name": "10 A4 sheets",
                    "item_name": "A4 paper",
                    "status": "fulfilled",
                    "inventory": {
                        "requested_quantity": 10,
                        "delivery_date": "2025-01-01",
                    },
                    "quote": {
                        "requested_quantity": 10,
                        "unit_price": 0.05,
                        "subtotal": 0.5,
                        "discount": 0.1,
                        "total_price": 0.45,
                    },
                    "fulfillment": {
                        "transactions": [
                            {"transaction_date": "2025-01-01"}
                        ]
                    },
                },
                {
                    "original_name": "200 specialty sheets",
                    "item_name": "Cardstock",
                    "status": "cannot_deliver_on_time",
                    "inventory": {
                        "requested_quantity": 200,
                        "delivery_date": "2025-01-15",
                    },
                },
                {
                    "original_name": (
                        "Paper for jane@example.com, call 212-555-0199"
                    ),
                    "item_name": None,
                    "status": "needs_clarification",
                    "message": (
                        "Please confirm the quantity with jane@example.com "
                        "or call 212-555-0199."
                    ),
                    "candidates": ["A4 paper", "Cardstock"],
                },
            ],
        }

        message = project.format_customer_response(response)
        summary = project.customer_status_summary(response)

        self.assertIn("How we calculated this:", message)
        self.assertIn("Catalog price: $0.05 x 10 = $0.50.", message)
        self.assertIn("Bulk discount: 10%.", message)
        self.assertNotIn("Reason code:", message)
        self.assertNotIn("ORDER_FULFILLED", message)
        self.assertNotIn("DELIVERY_WINDOW_UNAVAILABLE", message)
        self.assertNotIn("QTY_CLARIFICATION_REQUIRED", message)
        self.assertNotIn('{"', message)
        self.assertNotIn("jane@example.com", message)
        self.assertNotIn("212-555-0199", message)
        self.assertNotIn("private occasion", message)
        self.assertEqual(
            [item["reason_code"] for item in summary],
            [
                "ORDER_FULFILLED",
                "DELIVERY_WINDOW_UNAVAILABLE",
                "QTY_CLARIFICATION_REQUIRED",
            ],
        )

    def test_handle_request_stops_before_fulfillment_when_stock_is_short(self):
        agent = project.OrchestrationAgent(Mock())
        processed = {
            "items": [
                {
                    "original_name": "10 A4 sheets",
                    "item_name": "A4 paper",
                    "quantity": 10,
                    "requested_attributes": ["A4"],
                    "candidates": [],
                    "clarification_required": False,
                    "clarification_message": "",
                },
                {
                    "original_name": "200 balloons",
                    "item_name": None,
                    "quantity": 200,
                    "requested_attributes": [],
                    "candidates": [],
                    "clarification_required": True,
                    "clarification_message": "No matching catalog item.",
                },
            ],
            "request_date": "2025-01-01",
            "requested_delivery_date": "2025-01-04",
            "job": "manager",
            "event": "meeting",
            "needs_clarification": True,
        }
        inventory = {
            "item_name": "A4 paper",
            "exists": True,
            "requested_quantity": 10,
            "current_stock": 3,
            "enough_stock": False,
            "shortage": 7,
            "delivery_date": "2025-01-05",
            "order_will_be_fulfilled_on_time": False,
        }
        quote = {
            "item_name": "A4 paper",
            "requested_quantity": 10,
            "discount": 0,
            "total_price": 0.5,
            "pricing_explanation": "Base price 0.50 with no discount.",
        }

        with (
            patch.object(
                agent.tools["process_request"], "forward", return_value=processed
            ),
            patch.object(
                agent.tools["check_inventory"], "forward", return_value=inventory
            ) as check_inventory,
            patch.object(
                agent.tools["get_pricing"], "forward", return_value=quote
            ) as get_pricing,
            patch.object(
                agent.tools["finalize_order"], "forward"
            ) as finalize_order,
        ):
            result = agent.handle_request(
                "mixed request", "2025-01-01", "manager", "meeting"
            )

        self.assertTrue(result["needs_clarification"])
        check_inventory.assert_called_once()
        get_pricing.assert_not_called()
        finalize_order.assert_not_called()
        valid_item, blocked_item = result["items"]
        self.assertEqual(valid_item["status"], "cannot_deliver_on_time")
        self.assertEqual(blocked_item["status"], "needs_clarification")
        self.assertEqual(blocked_item["original_name"], "200 balloons")

    def test_delivery_date_gate_sends_on_time_shortage_to_order_agent(self):
        agent = project.OrchestrationAgent(Mock())
        processed = {
            "items": [
                {
                    "original_name": "10 A4 sheets",
                    "item_name": "A4 paper",
                    "quantity": 10,
                    "clarification_required": False,
                    "clarification_message": "",
                    "candidates": [],
                }
            ],
            "request_date": "2025-01-01",
            "requested_delivery_date": "2025-01-05",
            "job": "manager",
            "event": "meeting",
            "needs_clarification": False,
        }
        inventory = {
            "item_name": "A4 paper",
            "exists": True,
            "requested_quantity": 10,
            "current_stock": 3,
            "enough_stock": False,
            "shortage": 7,
            "delivery_date": "2025-01-05",
            "order_will_be_fulfilled_on_time": True,
        }
        quote = {
            "item_name": "A4 paper",
            "requested_quantity": 10,
            "discount": 0,
            "total_price": 0.5,
            "pricing_explanation": "Base price 0.50 with no discount.",
        }

        with (
            patch.object(
                agent.tools["process_request"], "forward", return_value=processed
            ),
            patch.object(
                agent.tools["check_inventory"], "forward", return_value=inventory
            ),
            patch.object(
                agent.tools["get_pricing"], "forward", return_value=quote
            ) as get_pricing,
            patch.object(
                agent.tools["finalize_order"],
                "forward",
                return_value={
                    "item_name": "A4 paper",
                    "requested_quantity": 10,
                    "transaction_type": "sales",
                    "status": "fulfilled",
                    "transactions": [],
                },
            ) as finalize_order,
        ):
            result = agent.handle_request(
                "shortage request", "2025-01-01", "manager", "meeting"
            )

        get_pricing.assert_called_once()
        finalize_order.assert_called_once()
        self.assertEqual(result["items"][0]["status"], "fulfilled")

    def test_can_fulfill_by_delivery_date_is_deterministic(self):
        on_time = {
            "exists": True,
            "enough_stock": False,
            "delivery_date": "2025-01-05",
        }
        late = {**on_time, "delivery_date": "2025-01-06"}
        self.assertTrue(
            project.can_fulfill_by_delivery_date(on_time, "2025-01-05")
        )
        self.assertFalse(
            project.can_fulfill_by_delivery_date(late, "2025-01-05")
        )

    def test_orchestrator_tools_delegate_and_return_structured_responses(self):
        agent = project.OrchestrationAgent(Mock())
        inventory_arguments = {
            "item_name": "A4 paper",
            "quantity": 10,
            "order_date": "2025-01-01",
        }
        pricing_arguments = {
            **inventory_arguments,
            "job": "manager",
            "event": "meeting",
        }
        order_arguments = {
            **inventory_arguments,
            "total_price": 0.45,
        }
        cases = [
            (
                "process_request",
                "request_processor_agent",
                {
                    "request": "I need 10 A4 sheets by 2025-01-10",
                    "request_date": "2025-01-01",
                    "job": "manager",
                    "event": "meeting",
                },
                {
                    "items": [
                        {
                            "original_name": "10 A4 sheets",
                            "item_name": "A4 paper",
                            "quantity": 10,
                            "requested_attributes": ["A4"],
                            "candidates": [],
                            "clarification_required": False,
                            "clarification_message": "",
                        }
                    ],
                    "request_date": "2025-01-01",
                    "requested_delivery_date": "2025-01-10",
                    "job": "manager",
                    "event": "meeting",
                    "needs_clarification": False,
                },
                ("I need 10 A4 sheets",),
            ),
            (
                "check_inventory",
                "inventory_agent",
                inventory_arguments,
                {
                    "item_name": "A4 paper",
                    "exists": True,
                    "requested_quantity": 10,
                    "current_stock": 3,
                    "enough_stock": False,
                    "shortage": 7,
                    "delivery_date": "2025-01-01",
                    "order_will_be_fulfilled_on_time": False,
                },
                ("A4 paper", "10"),
            ),
            (
                "get_pricing",
                "quote_agent",
                pricing_arguments,
                {
                    "item_name": "A4 paper",
                    "requested_quantity": 10,
                    "discount": 0.1,
                    "total_price": 0.45,
                    "unit_price": 0.05,
                    "subtotal": 0.5,
                    "pricing_explanation": "Bulk discount applied to the 0.50 subtotal.",
                },
                ("A4 paper", "10"),
            ),
            (
                "finalize_order",
                "order_agent",
                order_arguments,
                {
                    "item_name": "A4 paper",
                    "requested_quantity": 10,
                    "transaction_type": "sales",
                    "status": "fulfilled",
                    "transaction_date": "2025-01-01",
                },
                ("A4 paper", "10"),
            ),
        ]
        for tool_name, agent_name, arguments, response, expected_input in cases:
            for return_format, agent_response in (
                ("JSON string", json.dumps(response)),
                ("dictionary", response),
            ):
                with self.subTest(tool=tool_name, return_format=return_format):
                    specialist = getattr(agent, agent_name)
                    with patch.object(
                        specialist, "run", return_value=agent_response
                    ) as run:
                        result = agent.tools[tool_name](**arguments)
                    self.assertEqual(result, response)
                    run.assert_called_once()
                    prompt = run.call_args.args[0]
                    for value in expected_input:
                        self.assertIn(value, prompt)

    def test_orchestrator_tools_reject_invalid_specialist_responses(self):
        agent = project.OrchestrationAgent(Mock())
        cases = [
            (
                "process_request",
                "request_processor_agent",
                {
                    "request": "10 A4 sheets by 2025-01-10",
                    "request_date": "2025-01-01",
                    "job": "manager",
                    "event": "meeting",
                },
            ),
            (
                "check_inventory",
                "inventory_agent",
                {
                    "item_name": "A4 paper",
                    "quantity": 10,
                    "order_date": "2025-01-01",
                },
            ),
            (
                "get_pricing",
                "quote_agent",
                {
                    "item_name": "A4 paper",
                    "quantity": 10,
                    "order_date": "2025-01-01",
                    "job": "manager",
                    "event": "meeting",
                },
            ),
            (
                "finalize_order",
                "order_agent",
                {
                    "item_name": "A4 paper",
                    "quantity": 10,
                    "order_date": "2025-01-01",
                    "total_price": 0.45,
                },
            ),
        ]
        invalid_responses = [
            ("not JSON", json.JSONDecodeError),
            ("[]", TypeError),
            ([], TypeError),
            ("{}", ValueError),
            ({}, ValueError),
        ]
        for tool_name, agent_name, arguments in cases:
            for response, error_type in invalid_responses:
                with self.subTest(tool=tool_name, response=response):
                    specialist = getattr(agent, agent_name)
                    with patch.object(specialist, "run", return_value=response):
                        with self.assertRaises(error_type):
                            agent.tools[tool_name](**arguments)

    def test_validate_extracted_request_blocks_only_problematic_item(self):
        result = project.validate_extracted_request(
            {
                "items": [
                    {
                        "original_name": "10 A4 sheets",
                        "item_name": "A4 paper",
                        "quantity": 10,
                        "requested_attributes": ["A4"],
                        "candidates": [],
                        "clarification_required": False,
                        "clarification_message": "",
                    },
                    {
                        "original_name": "2 packs of specialty material",
                        "item_name": None,
                        "quantity": None,
                        "requested_attributes": [],
                        "candidates": ["Heavyweight paper", "250 gsm cardstock"],
                        "clarification_required": True,
                        "clarification_message": "Package size is missing.",
                    },
                ],
                "request_date": "2025-01-01",
                "requested_delivery_date": "2025-01-10",
                "job": "manager",
                "event": "meeting",
                "needs_clarification": False,
            }
        )

        valid_item, blocked_item = result["items"]
        self.assertFalse(valid_item["clarification_required"])
        self.assertTrue(blocked_item["clarification_required"])
        self.assertEqual(blocked_item["item_name"], None)
        self.assertIn("Package size is missing", blocked_item["clarification_message"])
        self.assertTrue(result["needs_clarification"])

    def test_parse_agent_value_accepts_one_extra_closing_brace(self):
        response = (
            '{"items": [], "order_date": "2025-04-04", '
            '"job": "manager", "event": "conference"}}'
        )
        self.assertEqual(
            project.parse_agent_value(response),
            {
                "items": [],
                "order_date": "2025-04-04",
                "job": "manager",
                "event": "conference",
            },
        )

    def test_parse_agent_value_rejects_non_brace_trailing_data(self):
        with self.assertRaises(json.JSONDecodeError):
            project.parse_agent_value(
                '{"order_date": "2025-04-04"} unexpected text'
            )

    def test_parse_agent_value_repairs_stray_apostrophe_before_brace(self):
        response = (
            '{"item_name":"Colored paper","requested_quantity":200,'
            '"discount":0.1,"pricing_explanation":"Final amount is 18.00.\'}'
        )
        self.assertEqual(
            project.parse_agent_value(response),
            {
                "item_name": "Colored paper",
                "requested_quantity": 200,
                "discount": 0.1,
                "pricing_explanation": "Final amount is 18.00.",
            },
        )

    def test_parse_agent_value_repairs_unescaped_dimension_quotes(self):
        response = (
            '{"original_name":"300 poster boards (24" x 36")",'
            '"item_name":"Large poster paper (24x36 inches)"}'
        )
        self.assertEqual(
            project.parse_agent_value(response),
            {
                "original_name": '300 poster boards (24" x 36")',
                "item_name": "Large poster paper (24x36 inches)",
            },
        )

    def test_parse_agent_value_repairs_quotes_and_extra_closing_brace(self):
        response = (
            '{"original_name":"300 poster boards (24" x 36")",'
            '"items":[]}}'
        )
        self.assertEqual(
            project.parse_agent_value(response),
            {
                "original_name": '300 poster boards (24" x 36")',
                "items": [],
            },
        )

    def test_parse_agent_value_unwraps_escaped_json_object(self):
        response = (
            '{\\n  \\"items\\": [],\\n  \\"order_date\\": \\"2025-04-04\\",\\n'
            '  \\"job\\": \\"unknown\\",\\n  \\"event\\": \\"conference\\"\\n}'
        )
        self.assertEqual(
            project.parse_agent_value(response),
            {
                "items": [],
                "order_date": "2025-04-04",
                "job": "unknown",
                "event": "conference",
            },
        )

    def test_parse_agent_value_unwraps_double_encoded_json_object(self):
        response = json.dumps(
            '{"items":[],"order_date":"2025-04-04"}'
        )
        self.assertEqual(
            project.parse_agent_value(response),
            {"items": [], "order_date": "2025-04-04"},
        )

    def test_validate_extracted_request_rejects_unknown_catalog_mapping(self):
        result = project.validate_extracted_request(
            {
                "items": [
                    {
                        "original_name": "heavy cardstock",
                        "item_name": "Heavy cardstock",
                        "quantity": 100,
                        "requested_attributes": ["heavy", "white"],
                        "candidates": ["Cardstock", "250 gsm cardstock"],
                        "clarification_required": False,
                        "clarification_message": "",
                    }
                ],
                "request_date": "2025-01-01",
                "requested_delivery_date": "2025-01-10",
                "job": "manager",
                "event": "meeting",
                "needs_clarification": False,
            }
        )

        item = result["items"][0]
        self.assertIsNone(item["item_name"])
        self.assertTrue(item["clarification_required"])
        self.assertIn("not an exact catalog item", item["clarification_message"])
        self.assertTrue(result["needs_clarification"])

    def test_minimum_stock_lookup(self):
        self.assertEqual(project.get_min_stock_level("A4 paper"), 2)
        self.assertEqual(
            project.get_min_stock_level("Paper cups"),
            project.DEFAULT_MIN_STOCK_LEVEL,
        )
        with self.assertRaisesRegex(ValueError, "Item not found"):
            project.get_min_stock_level("Not a catalog item")

    def test_reorder_rejects_nonpositive_quantity(self):
        with self.assertRaisesRegex(ValueError, "Quantity must be greater than zero"):
            project.reorder_item("A4 paper", 0, "2025-01-01", "2025-01-01")

    def test_order_agent_tool_reorders_shortage_and_restores_minimum_stock(self):
        stock = {
            "exists": True,
            "current_stock": 5,
            "shortage": 15,
            "min_stock_level": 2,
        }
        with patch.object(project, "get_stock_inventory", return_value=stock):
            result = project.reorder_item(
                "A4 paper", 20, "2025-01-01", "2025-01-01"
            )
        self.assertEqual(result["shortage_quantity"], 15)
        self.assertEqual(result["shortage_delivery_date"], "2025-01-02")
        self.assertEqual(result["safety_stock_quantity"], 2)
        self.assertEqual(result["safety_stock_delivery_date"], "2025-01-01")
        self.assertEqual(result["reordered_quantity"], 17)

    def test_execute_order_creates_separate_dated_transactions(self):
        stock = {
            "exists": True,
            "current_stock": 5,
            "shortage": 15,
        }
        with patch.object(project, "get_stock_inventory", return_value=stock):
            transactions = project.execute_order(
                "A4 paper", 20, "2025-01-01", 1.0
            )

        self.assertEqual(len(transactions), 3)
        self.assertEqual(
            [transaction["transaction_type"] for transaction in transactions],
            ["stock_orders", "stock_orders", "sales"],
        )
        self.assertEqual(
            [transaction["requested_quantity"] for transaction in transactions],
            [15, 2, 20],
        )
        self.assertEqual(
            [transaction["transaction_date"] for transaction in transactions],
            ["2025-01-02", "2025-01-01", "2025-01-02"],
        )

    def test_evaluate_workflow_response_records_structured_metrics(self):
        response = {
            "request_date": "2025-01-01",
            "requested_delivery_date": "2025-01-10",
            "items": [
                {
                    "original_name": "10 A4 sheets",
                    "item_name": "A4 paper",
                    "status": "fulfilled",
                    "quote": {
                        "item_name": "A4 paper",
                        "requested_quantity": 10,
                        "unit_price": 0.05,
                        "subtotal": 0.5,
                        "discount": 0.1,
                        "total_price": 0.45,
                    },
                    "fulfillment": {
                        "transactions": [
                            {
                                "transaction_id": 1,
                                "item_name": "A4 paper",
                                "transaction_type": "sales",
                                "transaction_date": "2025-01-01",
                            }
                        ]
                    },
                },
                {
                    "original_name": "balloons",
                    "item_name": None,
                    "status": "needs_clarification",
                },
            ],
        }

        evaluation = project.evaluate_workflow_response(response)

        self.assertTrue(evaluation["passed"])
        self.assertEqual(evaluation["item_count"], 2)
        self.assertEqual(evaluation["transaction_count"], 1)
        self.assertEqual(evaluation["items"][0]["discount_rate"], 0.1)
        self.assertEqual(
            evaluation["items"][0]["expected_arithmetic"]["expected_total"],
            0.45,
        )
        self.assertEqual(evaluation["items"][0]["transaction_ids"], [1])

    def test_evaluate_workflow_response_rejects_wrong_arithmetic(self):
        response = {
            "request_date": "2025-01-01",
            "requested_delivery_date": "2025-01-10",
            "items": [
                {
                    "original_name": "10 A4 sheets",
                    "item_name": "A4 paper",
                    "status": "fulfilled",
                    "quote": {
                        "item_name": "A4 paper",
                        "requested_quantity": 10,
                        "unit_price": 0.05,
                        "subtotal": 0.5,
                        "discount": 0.1,
                        "total_price": 0.5,
                    },
                    "fulfillment": {
                        "transactions": [
                            {
                                "transaction_id": 1,
                                "item_name": "A4 paper",
                                "transaction_type": "sales",
                                "transaction_date": "2025-01-01",
                            }
                        ]
                    },
                }
            ],
        }

        with self.assertRaisesRegex(AssertionError, "discount arithmetic"):
            project.evaluate_workflow_response(response)

    def test_execute_order_supports_catalog_item_without_inventory_record(self):
        transactions = project.execute_order(
            "Paper cups", 10, "2025-01-01", 0.8
        )

        self.assertEqual(
            [transaction["transaction_type"] for transaction in transactions],
            ["stock_orders", "stock_orders", "sales"],
        )
        self.assertEqual(
            [transaction["requested_quantity"] for transaction in transactions],
            [10, project.DEFAULT_MIN_STOCK_LEVEL, 10],
        )
        with self.engine.connect() as conn:
            inventory = conn.execute(
                text(
                    "SELECT unit_price, min_stock_level "
                    "FROM inventory WHERE item_name = 'Paper cups'"
                )
            ).one()
        self.assertAlmostEqual(inventory.unit_price, 0.08)
        self.assertEqual(
            inventory.min_stock_level, project.DEFAULT_MIN_STOCK_LEVEL
        )

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

    def test_supplier_delivery_date_rejects_invalid_request_date(self):
        with self.assertRaisesRegex(ValueError, "Invalid request date"):
            project.get_supplier_delivery_date("not-a-date", 10)

    def test_quote_history_respects_request_date(self):
        with self.engine.begin() as conn:
            conn.execute(
                text("""
                INSERT INTO quote_requests (id, response)
                VALUES (2, 'Future A4 paper request')
                """)
            )
            conn.execute(
                text("""
                INSERT INTO quotes (
                    request_id, total_amount, quote_explanation, job_type,
                    order_size, event_type, order_date
                )
                VALUES (
                    2, 20.0, 'Future A4 paper quote', 'manager',
                    'large', 'conference', '2025-01-03'
                )
                """)
            )

        history = project.search_quote_history(
            ["A4 paper"], as_of_date="2025-01-01"
        )
        self.assertEqual([quote["total_amount"] for quote in history], [10.0])

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
        quotes = pd.DataFrame(
            [
                {
                    "total_amount": 10.0,
                    "quote_explanation": "A4 paper quote",
                    "request_metadata": (
                        "{'job_type': 'office manager', 'order_size': 'small', "
                        "'event_type': 'meeting'}"
                    ),
                }
            ]
        )
        engine = create_engine("sqlite:///:memory:")
        try:
            with patch.object(project.pd, "read_csv", side_effect=[requests, quotes]):
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


@unittest.skipUnless(
    os.getenv("RUN_LLM_INTEGRATION_TESTS") == "1",
    "Set RUN_LLM_INTEGRATION_TESTS=1 to run tests against the real LLM",
)
@unittest.skipUnless(
    os.getenv("UDACITY_OPENAI_API_KEY"),
    "UDACITY_OPENAI_API_KEY is required for real-LLM integration tests",
)
class LLMIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.original_engine = project.db_engine
        self.engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        project.db_engine = self.engine

        with self.engine.begin() as conn:
            conn.execute(
                text("""
                CREATE TABLE transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    item_name TEXT,
                    transaction_type TEXT,
                    units INTEGER,
                    price REAL,
                    transaction_date TEXT
                )
                """)
            )
            conn.execute(
                text("""
                CREATE TABLE inventory (
                    item_name TEXT,
                    category TEXT,
                    unit_price REAL,
                    current_stock INTEGER,
                    min_stock_level INTEGER
                )
                """)
            )
            conn.execute(
                text("""
                CREATE TABLE quote_requests (
                    id INTEGER,
                    response TEXT
                )
                """)
            )
            conn.execute(
                text("""
                CREATE TABLE quotes (
                    request_id INTEGER,
                    total_amount REAL,
                    quote_explanation TEXT,
                    job_type TEXT,
                    order_size TEXT,
                    event_type TEXT,
                    order_date TEXT
                )
                """)
            )
            conn.execute(
                text("""
                INSERT INTO inventory
                    (item_name, category, unit_price, current_stock, min_stock_level)
                VALUES ('A4 paper', 'paper', 0.05, 5, 2)
                """)
            )
            conn.execute(
                text("""
                INSERT INTO transactions
                    (item_name, transaction_type, units, price, transaction_date)
                VALUES
                    (NULL, 'sales', 1, 100.0, '2025-01-01'),
                    ('A4 paper', 'stock_orders', 5, 0.25, '2025-01-01')
                """)
            )

        self.agent = project.OrchestrationAgent(project.model)
        self.request_information = {
            "item_name": "A4 paper",
            "quantity": 3,
            "order_date": "2025-01-01",
            "job": "office manager",
            "event": "meeting",
            "total_price": 0.15,
        }

    def tearDown(self):
        project.db_engine = self.original_engine
        self.engine.dispose()

    def test_process_request_returns_requested_format(self):
        response = self.agent.tools["process_request"](
            request=(
                "I need 10 A4 paper for a meeting. "
                "Please deliver by 2025-01-10."
            ),
            request_date="2025-01-01",
            job="office manager",
            event="meeting",
        )

        self.assertIsInstance(response, dict)

        expected_fields = {
            "items",
            "request_date",
            "requested_delivery_date",
            "job",
            "event",
            "needs_clarification",
        }
        self.assertEqual(set(response), expected_fields)
        self.assertFalse(response["needs_clarification"])
        self.assertEqual(len(response["items"]), 1)
        self.assertEqual(response["items"][0]["item_name"], "A4 paper")
        self.assertEqual(response["items"][0]["quantity"], 10)
        self.assertEqual(response["request_date"], "2025-01-01")
        self.assertEqual(response["requested_delivery_date"], "2025-01-10")
        self.assertEqual(response["job"], "office manager")
        self.assertEqual(response["event"], "meeting")

    def test_check_inventory_returns_requested_format(self):
        response = self.agent.tools["check_inventory"](
            item_name="A4 paper",
            quantity=3,
            order_date="2025-01-01",
        )

        expected_fields = {
            "item_name",
            "exists",
            "requested_quantity",
            "current_stock",
            "enough_stock",
            "shortage",
            "delivery_date",
        }
        self.assertIsInstance(response, dict)
        self.assertEqual(set(response), expected_fields)
        self.assertEqual(response["item_name"], "A4 paper")
        self.assertEqual(response["requested_quantity"], 3)
        self.assertEqual(response["current_stock"], 5)
        self.assertTrue(response["exists"])
        self.assertTrue(response["enough_stock"])
        self.assertEqual(response["shortage"], 0)

    def test_get_pricing_returns_requested_format(self):
        response = self.agent.tools["get_pricing"](
            item_name="A4 paper",
            quantity=3,
            order_date="2025-01-01",
            job="office manager",
            event="meeting",
        )

        expected_fields = {
            "item_name",
            "requested_quantity",
            "discount",
            "total_price",
            "pricing_explanation",
            "unit_price",
            "subtotal",
        }
        self.assertIsInstance(response, dict)
        self.assertEqual(set(response), expected_fields)
        self.assertEqual(response["item_name"], "A4 paper")
        self.assertEqual(response["requested_quantity"], 3)
        self.assertGreaterEqual(response["discount"], 0)
        self.assertLessEqual(response["discount"], 1)
        self.assertGreater(response["total_price"], 0)
        self.assertAlmostEqual(
            response["total_price"],
            round(response["subtotal"] * (1 - response["discount"]), 2),
        )
        self.assertIsInstance(response["pricing_explanation"], str)
        self.assertTrue(response["pricing_explanation"].strip())

    def test_finalize_order_returns_requested_format(self):
        response = self.agent.tools["finalize_order"](
            item_name="A4 paper",
            quantity=3,
            order_date="2025-01-01",
            total_price=0.15,
        )

        expected_fields = {
            "item_name",
            "requested_quantity",
            "transaction_type",
            "status",
            "transactions",
        }
        self.assertIsInstance(response, dict)
        self.assertEqual(set(response), expected_fields)
        self.assertEqual(response["item_name"], "A4 paper")
        self.assertEqual(response["requested_quantity"], 3)
        self.assertIn(response["transaction_type"], {"sales", "multiple"})
        self.assertTrue(response["status"])
        for transaction in response.get("transactions", []):
            self.assertIn("transaction_date", transaction)

if __name__ == "__main__":
    unittest.main()
