"""
DeepMyst MCP Server with SSE and STDIO Support

This server provides access to DeepMyst optimization and routing capabilities through MCP.
It supports both STDIO transport (for Claude Desktop) and SSE transport (for HTTP clients).
API keys are provided by clients during tool calls rather than stored on the server.

Usage:
- For Claude Desktop: python deepmyst_mcp.py --stdio
- For HTTP clients: python deepmyst_mcp.py --sse
"""

import os
import sys
import json
import argparse
import aiohttp
import logging
from typing import Dict, List, Any, Optional
from openai import OpenAI
from mcp.server.fastmcp import FastMCP, Context
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.responses import JSONResponse
from starlette.endpoints import HTTPEndpoint
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("deepmyst.log"),
        logging.StreamHandler(sys.stderr)  # Use stderr to avoid contaminating stdout in STDIO mode
    ]
)
logger = logging.getLogger("deepmyst_mcp")

# Parse command line arguments properly
def parse_args():
    parser = argparse.ArgumentParser(description='DeepMyst MCP Server')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--stdio', action='store_true', help='Run with STDIO transport (default for Claude Desktop)')
    group.add_argument('--sse', action='store_true', help='Run with SSE transport over HTTP')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind SSE server to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind SSE server to')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    # Parse known args only, to avoid conflicts with other libraries
    args, unknown = parser.parse_known_args()
    return args

# Detect if running in stdio mode from environment or command line
def is_stdio_mode():
    """Determine if the server should run in STDIO mode"""
    # Check command line args
    if len(sys.argv) > 1 and sys.argv[1] == "--stdio":
        return True
    
    # Check environment variable as backup
    if os.environ.get("MCP_TRANSPORT", "").lower() == "stdio":
        return True
    
    # Check if being run by MCP Inspector (usually indicates stdio mode)
    if os.environ.get("MCP_INSPECTOR"):
        return True
    
    return False

# Initialize FastMCP server
mcp = FastMCP("DeepMyst", version="1.0.0")

# DeepMyst Router API URL
ROUTER_URL = "https://router.deepmyst.com/route"

# Default API request timeout
DEFAULT_TIMEOUT = 50

# Model provider mappings
PROVIDER_MODEL_MAP = {
    # OpenAI models
    "gpt-4o": "openai",
    "gpt-4o-mini": "openai",
    "o1": "openai",
    "o1-mini": "openai",
    "o3-mini": "openai",
    "chatgpt-4o-latest": "openai",
    
    # Anthropic models
    "claude-3-7-sonnet-20250219": "claude",
    "claude-3-5-sonnet-latest": "claude",
    "claude-3-5-haiku-latest": "claude",
    "claude-3-opus-latest": "claude",
    "claude-3-7-sonnet": "claude",
    "claude-3-5-sonnet": "claude",
    "claude-3-haiku": "claude", 
    "claude-3-opus": "claude",
    
    # Google models
    "gemini-2.0-flash": "gemini",
    "gemini-2.0-flash-lite-preview-02-05": "gemini",
    "gemini-1.5-pro": "gemini",
    "gemini-1.5-flash": "gemini",
    "gemini-1.5-flash-8b": "gemini",
    
    # Groq models
    "llama-3.1-8b-instant": "groq",
    "llama-3.3-70b-versatile": "groq", 
    "llama-guard-3-8b": "groq",
    "mixtral-8x7b-32768": "groq",
    "gemma2-9b-it": "groq",
    "qwen-2.5-32b": "groq",
    "deepseek-r1-distill-qwen-32b": "groq",
    "deepseek-r1-distill-llama-70b": "groq",
    "llama-3.1": "groq",
    "llama-3.3": "groq",
    "mixtral-8x7b": "groq",
    "gemma2": "groq",
    "qwen": "groq",
    "deepseek": "groq"
}

# Configure OpenAI client with DeepMyst endpoint
def get_client(api_key=None):
    """Get configured OpenAI client with DeepMyst endpoint

    Args:
        api_key: The DeepMyst API key (required if not set as environment variable)
    
    Returns:
        OpenAI client configured to use DeepMyst endpoint
    
    Raises:
        ValueError: If no API key is provided
    """
    # First try to get API key from parameter
    if not api_key:
        # Fall back to environment variable as a backup
        api_key = os.environ.get("DEEPMYST_API_KEY")
        
    if not api_key:
        raise ValueError("No DeepMyst API key provided. Please provide a key as a parameter.")
    
    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepmyst.com/v1"
    )

# Helper function to prepare messages consistently
def prepare_messages(prompt, system_message=None):
    """Prepare messages for LLM completion in a consistent format

    Args:
        prompt: The user prompt text
        system_message: Optional system message to include

    Returns:
        List of message objects for the API call
    """
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    
    messages.append({"role": "user", "content": prompt})
    return messages

@mcp.tool()
async def get_best_model_for_query(
    query: str,
    api_key: str,  # Required API key parameter
    require_reasoning: bool = False,
    fast_response: bool = False,
    cost_sensitive: bool = False,
    ctx: Context = None
) -> str:
    """Determine the best LLM model for a specific query using DeepMyst's router.
    
    Args:
        query: The text of the query to analyze
        api_key: Your DeepMyst API key
        require_reasoning: Whether the query requires strong reasoning capabilities
        fast_response: Whether response speed is a priority
        cost_sensitive: Whether to prioritize lower-cost models
        ctx: MCP context object
    
    Returns:
        JSON object with the recommended model and details
    """
    try:
        if not api_key:
            error_msg = "DeepMyst API key is required"
            if ctx:
                ctx.error(error_msg)
            return {"error": error_msg}
        
        # Create payload for router API
        payload = {
            "query_text": query
        }
        
        # Add optional parameters if specified
        if require_reasoning:
            payload["intelligence_priority"] = 0.9
        if fast_response:
            payload["speed_priority"] = 0.9
        if cost_sensitive:
            payload["cost_priority"] = 0.9
            
        if ctx:
            ctx.info(f"Calling DeepMyst router API with payload: {payload}")
        
        # Call the router API
        async with aiohttp.ClientSession() as session:
            headers = {
                'Content-Type': 'application/json',
                'X-API-Key': api_key  # Use provided API key
            }
            
            try:
                async with session.post(ROUTER_URL, headers=headers, json=payload, timeout=DEFAULT_TIMEOUT) as response:
                    if response.status == 200:
                        routing_data = await response.json()
                        if ctx:
                            ctx.info(f"Router API response: {routing_data}")
                        
                        # Parse and enhance the response
                        result = {
                            "recommended_model": routing_data.get("selected_llm", "gpt-4o"),
                            "provider": PROVIDER_MODEL_MAP.get(routing_data.get("selected_llm", "gpt-4o"), "unknown"),
                            "model_scores": routing_data.get("model_scores", {}),
                            "query_analysis": routing_data.get("query_analysis", {}),
                            "fallback_model": "gpt-4o"  # Always include a fallback model
                        }
                        
                        return json.dumps(result)
                    else:
                        error_text = await response.text()
                        error_msg = f"Router API error: {response.status} - {error_text}"
                        if ctx:
                            ctx.error(error_msg)
                        return json.dumps({"error": error_msg, "fallback_model": "gpt-4o"})
            except Exception as e:
                error_msg = f"Network error when calling router API: {str(e)}"
                if ctx:
                    ctx.error(error_msg)
                return json.dumps({"error": error_msg, "fallback_model": "gpt-4o"})
                
    except Exception as e:
        error_msg = f"Exception when calling router API: {str(e)}"
        if ctx:
            ctx.error(error_msg)
        return json.dumps({"error": error_msg, "fallback_model": "gpt-4o"})

@mcp.tool()
async def optimized_completion(
    prompt: str,
    api_key: str,  # Required API key parameter
    model: str = "gpt-4o-mini",
    system_message: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    ctx: Context = None
) -> str:
    """Generate an LLM response using DeepMyst's token optimization given a specific model name.
    Use this to talk to other LLMs and agents.
    
    Args:
        prompt: The user prompt to send to the model
        api_key: Your DeepMyst API key
        model: Base model to use (will add -optimize flag)
        system_message: Optional system message
        temperature: Controls randomness (0-1)
        max_tokens: Maximum tokens in the response
        ctx: MCP context object
    
    Returns:
        The model's response text
    """
    try:
        if not api_key:
            error_msg = "DeepMyst API key is required"
            if ctx:
                ctx.error(error_msg)
            return f"Error: DeepMyst API key is required"
            
        client = get_client(api_key)  # Use provided API key
        
        # Add optimization flag
        optimized_model = f"{model}-optimize"
        
        # Prepare messages consistently
        messages = prepare_messages(prompt, system_message)
        
        # Call DeepMyst API
        if ctx:
            ctx.info(f"Calling DeepMyst with optimized model: {optimized_model}")
        response = client.chat.completions.create(
            model=optimized_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=DEFAULT_TIMEOUT
        )
        
        return response.choices[0].message.content
    except Exception as e:
        error_msg = f"Error in optimized_completion: {str(e)}"
        if ctx:
            ctx.error(error_msg)
        return f"An error occurred: {str(e)}"

@mcp.tool()
async def auto_routed_completion(
    prompt: str,
    api_key: str,  # Required API key parameter
    base_model: str = "gpt-4o-mini",
    system_message: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    ctx: Context = None
) -> str:
    """Generate an LLM response using DeepMyst's smart routing. 
    Use this to find an LLM that can answer a question or prompt for you.
    
    Args:
        prompt: The user prompt to send to the model
        api_key: Your DeepMyst API key
        base_model: Base model to use (will add -auto flag)
        system_message: Optional system message
        temperature: Controls randomness (0-1)
        max_tokens: Maximum tokens in the response
        ctx: MCP context object
    
    Returns:
        The model's response text
    """
    try:
        if not api_key:
            error_msg = "DeepMyst API key is required"
            if ctx:
                ctx.error(error_msg)
            return f"Error: DeepMyst API key is required"
            
        client = get_client(api_key)  # Use provided API key
        
        # Add auto routing flag
        routed_model = f"{base_model}-auto"
        
        # Prepare messages consistently
        messages = prepare_messages(prompt, system_message)
        
        # Call DeepMyst API
        if ctx:
            ctx.info(f"Calling DeepMyst with auto-routed model: {routed_model}")
        response = client.chat.completions.create(
            model=routed_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=DEFAULT_TIMEOUT
        )
        
        return response.choices[0].message.content
    except Exception as e:
        error_msg = f"Error in auto_routed_completion: {str(e)}"
        if ctx:
            ctx.error(error_msg)
        return f"An error occurred: {str(e)}"

@mcp.tool()
async def deepmyst_completion(
    prompt: str,
    api_key: str,  # Required API key parameter
    base_model: str = "gpt-4o-mini",
    system_message: Optional[str] = None,
    optimize: bool = True,
    auto_route: bool = True,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    ctx: Context = None
) -> str:
    """Generate an LLM response using DeepMyst's features.
    
    Args:
        prompt: The user prompt to send to the model
        api_key: Your DeepMyst API key
        base_model: Base model to use
        system_message: Optional system message
        optimize: Whether to enable token optimization
        auto_route: Whether to enable smart routing
        temperature: Controls randomness (0-1)
        max_tokens: Maximum tokens in the response
        ctx: MCP context object
    
    Returns:
        The model's response text
    """
    try:
        if not api_key:
            error_msg = "DeepMyst API key is required"
            if ctx:
                ctx.error(error_msg)
            return f"Error: DeepMyst API key is required"
            
        client = get_client(api_key)  # Use provided API key
        
        # Build model name with flags
        model_name = base_model
        if auto_route:
            model_name += "-auto"
        if optimize:
            model_name += "-optimize"
        
        # Prepare messages consistently
        messages = prepare_messages(prompt, system_message)
        
        # Call DeepMyst API
        if ctx:
            ctx.info(f"Calling DeepMyst with model: {model_name}")
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=DEFAULT_TIMEOUT
        )
        
        return response.choices[0].message.content
    except Exception as e:
        error_msg = f"Error in deepmyst_completion: {str(e)}"
        if ctx:
            ctx.error(error_msg)
        return f"An error occurred: {str(e)}"

@mcp.tool()
async def smart_completion(
    prompt: str,
    api_key: str,  # Required API key parameter
    system_message: Optional[str] = None,
    optimize: bool = True,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    ctx: Context = None
) -> str:
    """Generate a response using the best model for the query, chosen by DeepMyst's router.
    Use this to get help from other LLMs to answer a user query.
    
    Args:
        prompt: The user prompt to send to the model
        api_key: Your DeepMyst API key
        system_message: Optional system message
        optimize: Whether to enable token optimization
        temperature: Controls randomness (0-1)
        max_tokens: Maximum tokens in the response
        ctx: MCP context object
    
    Returns:
        The model's response text
    """
    try:
        if not api_key:
            error_msg = "DeepMyst API key is required"
            if ctx:
                ctx.error(error_msg)
            return f"Error: DeepMyst API key is required"
            
        client = get_client(api_key)  # Use provided API key
        
        # First, get the best model for this query using the router
        router_response_str = await get_best_model_for_query(prompt, api_key, ctx=ctx)
        router_response = json.loads(router_response_str)
        
        # Check if the router call was successful
        if isinstance(router_response, dict) and "error" in router_response:
            # Use fallback if there's an error
            selected_model = router_response.get("fallback_model", "gpt-4o")
            if ctx:
                ctx.warning(f"Router API call failed, using fallback model: {selected_model}")
        else:
            # Use the recommended model from the router
            selected_model = router_response.get("recommended_model", "gpt-4o")
            if ctx:
                ctx.info(f"Using router-recommended model: {selected_model}")
        
        # Add optimization flag if requested
        if optimize:
            selected_model += "-optimize"
        
        # Prepare messages consistently
        messages = prepare_messages(prompt, system_message)
        
        # Call DeepMyst API
        if ctx:
            ctx.info(f"Calling DeepMyst with model: {selected_model}")
        response = client.chat.completions.create(
            model=selected_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=DEFAULT_TIMEOUT
        )
        
        return response.choices[0].message.content
    except Exception as e:
        error_msg = f"Error in smart_completion: {str(e)}"
        if ctx:
            ctx.error(error_msg)
        return f"An error occurred: {str(e)}"

# Add information resource
@mcp.resource("deepmyst://info")
def get_deepmyst_info() -> str:
    """Get information about DeepMyst's capabilities"""
    return """
    DeepMyst provides two powerful capabilities:
    
    1. Token Optimization: Reduces token usage by up to 75% while preserving response quality.
    
    2. Smart Routing: Automatically selects the optimal model based on query characteristics.
    
    3. Model Gateway: Access other LLMs including GPT-4o-mini and other models which can help you solve problems and collaborate with you.
  
    Combine the above tools to get a team to help you address user needs and prompts. Like collaborating with other LLMs, optimizing user queries and prompts. Finding the best LLM to answer the question or to validate your own answers using other LLMs.
    
    IMPORTANT: You must provide your DeepMyst API key when calling any tool. Get your API key from https://platform.deepmyst.com.
    
    Example usage:
    
    Use smart_completion to explain quantum computing, with these parameters:
    - api_key: your-deepmyst-api-key
    - optimize: true
    """

# API health check endpoint
@mcp.resource("deepmyst://health")
def get_health_check() -> str:
    """Get server health status"""
    try:
        return json.dumps({
            "status": "healthy",
            "version": "1.0.0",
            "server": "deepmyst_mcp",
            "transport": "SSE and STDIO supported",
            "api_key_required": True
        })
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": str(e)
        })

# Home page handler for SSE mode
class HomepageEndpoint(HTTPEndpoint):
    async def get(self, request):
        return JSONResponse({
            "name": "DeepMyst MCP Server",
            "version": "1.0.0", 
            "status": "running",
            "endpoints": {
                "/": "API documentation",
                "/sse": "SSE connection endpoint",
                "/messages": "HTTP POST endpoint for messages"
            },
            "documentation": "For more information, visit https://platform.deepmyst.com"
        })

# Run the server
if __name__ == "__main__":
    try:
        # Parse command line arguments
        args = parse_args()
        
        # Set up additional logging if debug mode is enabled
        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.setLevel(logging.DEBUG)
            logger.debug("Debug logging enabled")
        
        # Determine which transport to use
        use_stdio = args.stdio or (not args.sse and is_stdio_mode())
        
        if use_stdio:
            # Run with stdio transport (for Claude Desktop or MCP Inspector)
            logger.info("Starting DeepMyst MCP server with stdio transport")
            # Always use stderr for console output in STDIO mode to avoid contaminating the transport
            print("Starting DeepMyst MCP server with stdio transport", file=sys.stderr)
            print("API keys must be provided with each tool call", file=sys.stderr)
            mcp.run(transport='stdio')
        else:
            # Run with SSE transport (for HTTP clients)
            host = args.host or os.environ.get("HOST", "0.0.0.0")
            port = args.port or int(os.environ.get("PORT", 8000))
            
            logger.info(f"Starting DeepMyst MCP server with SSE transport on {host}:{port}")
            print(f"Starting DeepMyst MCP server with SSE transport on {host}:{port}")
            print("API keys must be provided with each tool call")
            
            # Use the FastMCP sse_app method to create a proper ASGI application
            sse_app = mcp.sse_app()
            
            # Configure Starlette app with homepage endpoint and mount the SSE app
            app = Starlette(routes=[
                Route("/", HomepageEndpoint),
            ])
            
            # Mount the SSE app at the root
            from starlette.routing import Mount
            app.routes.append(Mount("/", app=sse_app))
            
            # Run the server with uvicorn
            uvicorn.run(app, host=host, port=port)
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        print(f"Error starting DeepMyst MCP server: {str(e)}", file=sys.stderr)
        sys.exit(1)