#!/usr/bin/env python
"""Minimal MCP server for debugging - this uses stdio for communication."""
import sys 
import asyncio
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types
#All logs to stderr
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)],
    force=True
)
logger=logging.getLogger(__name__)

server=Server("math-server")

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    logger.info("list_tools called")
    return [
        types.Tool(
            name="add",
            description="Add two numbers",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {"type":"number"},
                    "b": {"type": "number"}
                },
                "required":["a","b"]
            }
        ),
        types.Tool(
            name="subtract",
            description="Subtract two numbers",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {"type":"number"},
                    "b": {"type": "number"}
                },
                "required":["a","b"]
            }
        ),
        types.Tool(
            name="Multiply",
            description="Multiply two numbers",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {"type":"number"},
                    "b": {"type": "number"}
                },
                "required":["a","b"]
            }
        ),
        types.Tool(
            name="Divide",
            description="Divide two numbers",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {"type":"number"},
                    "b": {"type": "number"}
                },
                "required":["a","b"]
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name:str, arguments: dict) -> list[types.TextContent]:
    logger.info(f"call_tool: {name}")
    '''if name == "add":
        result=arguments["a"]+arguments["b"]
        return [types.TextContent(type="text",text=str(result))]
    elif name == "subtract":
        result=arguments["a"]-arguments["b"]
        return [types.TextContent(type="text",text=str(result))]
    elif name == "Multiply":
        result=arguments["a"]*arguments["b"]
        return [types.TextContent(type="text",text=str(result))]
    raise ValueError(f"Unknown tool: {name}")'''

    match name:
        case "add":
            result = arguments["a"] + arguments["b"]
            return [types.TextContent(type="text", text=str(result))]
        case "subtract":
            result = arguments["a"] - arguments["b"]
            return [types.TextContent(type="text", text=str(result))]
        case "Multiply":
            result = arguments["a"] * arguments["b"]
            return [types.TextContent(type="text", text=str(result))]
        case "Divide":
            result = arguments["a"] // arguments["b"]
            return [types.TextContent(type="text", text=str(result))]
        case _:
            raise ValueError(f"Unknown tool: {name}")


async def main():
    logger.info("Starting server...")
    async with stdio_server() as (read_stream,write_stream):
        logger.info("stdio_server ready")
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__=="__main__":
    asyncio.run(main())