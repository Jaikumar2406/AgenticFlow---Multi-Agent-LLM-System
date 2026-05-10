import asyncio
from typing import Dict, Any, List, Optional
from .base import BaseTool
import hashlib
import re


class DataLookupTool(BaseTool):
    """Structured data lookup with natural language to SQL conversion"""

    def __init__(self):
        super().__init__("data_lookup", timeout=5)
        # Mock database for demo
        self._mock_data = {
            "employees": [
                {"id": 1, "name": "Alice Johnson", "department": "Engineering", "salary": 120000},
                {"id": 2, "name": "Bob Smith", "department": "Sales", "salary": 90000},
                {"id": 3, "name": "Carol Williams", "department": "Engineering", "salary": 110000},
                {"id": 4, "name": "David Brown", "department": "Marketing", "salary": 85000},
                {"id": 5, "name": "Eve Davis", "department": "Engineering", "salary": 130000}
            ],
            "products": [
                {"id": 1, "name": "Widget A", "category": "Electronics", "price": 99.99, "stock": 150},
                {"id": 2, "name": "Widget B", "category": "Electronics", "price": 149.99, "stock": 75},
                {"id": 3, "name": "Gadget X", "category": "Tools", "price": 49.99, "stock": 200},
                {"id": 4, "name": "Gadget Y", "category": "Tools", "price": 79.99, "stock": 50}
            ]
        }
        self._sql_patterns = {
            r"list (all )?(\w+)": "SELECT * FROM {table}",
            r"how many (\w+)": "SELECT COUNT(*) as count FROM {table}",
            r"average (salary|price|stock)": "SELECT AVG({col}) as avg FROM {table}",
            r"(filter|where) (.*)": "SELECT * FROM {table} WHERE {condition}",
        }

    def get_failure_contract(self) -> Dict[str, str]:
        return {
            "timeout": "{'success': false, 'error': 'Query timeout after 5s', 'data': null}",
            "empty_results": "{'success': true, 'data': [], 'sql_query': null, 'error': null}",
            "malformed_input": "{'success': false, 'error': 'Invalid input: query must be a non-empty string', 'data': null}"
        }

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data lookup"""
        # Validate input
        if not isinstance(input_data, dict):
            return {
                "success": False,
                "error": "Invalid input: must be a dictionary",
                "data": None
            }

        query = input_data.get("query", "")
        if not query or not isinstance(query, str):
            return {
                "success": False,
                "error": "Invalid input: query must be a non-empty string",
                "data": None
            }

        # Simulate query processing delay
        await asyncio.sleep(0.1)

        # Convert natural language to SQL-like operation
        sql_query, table, condition = self._parse_query(query)

        if not table:
            return {
                "success": False,
                "error": "Could not determine data table from query",
                "data": None
            }

        # Execute query on mock data
        data = self._mock_data.get(table, [])
        result = self._execute_query(data, sql_query, condition)

        # Generate result hash
        result_hash = hashlib.sha256(str(result).encode()).hexdigest()[:16]

        return {
            "success": True,
            "data": {
                "query": query,
                "sql_query": sql_query,
                "results": result,
                "table": table,
                "row_count": len(result)
            },
            "result_hash": result_hash,
            "error": None
        }

    def _parse_query(self, query: str) -> tuple:
        """Parse natural language to SQL-like operation"""
        query_lower = query.lower()

        # Determine table
        table = None
        for tbl in self._mock_data.keys():
            if tbl in query_lower:
                table = tbl
                break

        if not table:
            # Default to employees
            table = "employees"

        # Determine operation
        if "how many" in query_lower:
            return "COUNT", table, None
        elif "average" in query_lower:
            return "AVG", table, None
        elif "filter" in query_lower or "where" in query_lower:
            # Extract simple condition
            return "FILTER", table, query_lower
        else:
            return "SELECT", table, None

    def _execute_query(self, data: List[Dict], operation: str, condition: Optional[str]) -> Any:
        """Execute the parsed query on data"""
        if operation == "COUNT":
            return [{"count": len(data)}]
        elif operation == "AVG":
            # Determine which column to average
            col = "salary" if "salary" in (condition or "") else "price"
            if data and col in data[0]:
                avg = sum(d.get(col, 0) for d in data) / len(data)
                return [{"avg": round(avg, 2)}]
            return [{"avg": 0}]
        elif operation == "FILTER":
            # Simple filtering for demo
            return data[:3]  # Return first 3 for any filter
        else:
            return data


data_lookup_tool = DataLookupTool()