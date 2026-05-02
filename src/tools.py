import pandas as pd
from llama_index.core.tools import QueryEngineTool, ToolMetadata, FunctionTool
from src.config import SIMILARITY_TOP_K


def create_query_engine_tool(index, llm):
    query_engine = index.as_query_engine(similarity_top_k=SIMILARITY_TOP_K, llm=llm)
    return QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="doc_search",
            description="Provides information about the company's finances. Use whenever user asks for something.",
        ),
    )


def create_pandas_tool(df: pd.DataFrame, llm):
    from llama_index.experimental.query_engine.pandas.pandas_query_engine import PandasQueryEngine

    pandas_engine = PandasQueryEngine(df=df, llm=llm, verbose=True)

    def query_spreadsheet(query: str) -> str:
        """Queries the already loaded CSV spreadsheet with financial data (revenues, expenses, dates, categories). Use this tool for ANY question about the data, finances, revenues, expenses, hypotheses or analysis. The data is already available - just pass the user's question."""
        return str(pandas_engine.query(query))

    return FunctionTool.from_defaults(fn=query_spreadsheet), pandas_engine


def multiply(a: float, b: float) -> float:
    """Multiply two values and returns the result"""
    return a * b


def add(a: float, b: float) -> float:
    """Add two values and returns the result"""
    return a + b


def subtract(a: float, b: float) -> float:
    """Subtract two values and returns the result"""
    return a - b


def divide(a: float, b: float) -> float:
    """Divides two values and returns the result"""
    return a / b


def growth_rate(previous: float, current: float) -> float:
    """Useful for calculating the percentage growth rate between two values."""
    return ((current - previous) / previous) * 100


def get_math_tools():
    return [
        FunctionTool.from_defaults(fn=multiply),
        FunctionTool.from_defaults(fn=add),
        FunctionTool.from_defaults(fn=subtract),
        FunctionTool.from_defaults(fn=divide),
        FunctionTool.from_defaults(fn=growth_rate),
    ]
