"""
DeepMyst MCP Server with SSE and STDIO Support

This server provides access to DeepMyst optimization and routing capabilities through MCP.
It supports both STDIO transport (for Claude Desktop) and SSE transport (for HTTP clients).

Usage:
- For Claude Desktop: python deepmyst_mcp.py --stdio
- For HTTP clients: python deepmyst_mcp.py
"""

import os
import sys
import json
import aiohttp
import logging
from typing import Dict, List, Any, Optional
from openai import OpenAI
from mcp.server.fastmcp import FastMCP, Context
from mcp.server.sse import SseServerTransport
import uvicorn
from starlette.applications import Starlette
from starlette.routing import Route

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("deepmyst.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("deepmyst_mcp")

# Initialize FastMCP server
mcp = FastMCP("DeepMyst")

# DeepMyst Router API URL
ROUTER_URL = "https://router.deepmyst.com/route"

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
def get_client():
    """Get configured OpenAI client with DeepMyst endpoint"""
    api_key = os.environ.get("DEEPMYST_API_KEY")
    if not api_key:
        raise ValueError("DEEPMYST_API_KEY environment variable not set")
    
    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepmyst.com/v1"
    )

@mcp.tool()
async def get_best_model_for_query(
    query: str,
    require_reasoning: bool = False,
    fast_response: bool = False,
    cost_sensitive: bool = False,
    ctx: Context = None
) -> str:
    """Determine the best LLM model for a specific query using DeepMyst's router.
    
    Args:
        query: The text of the query to analyze
        require_reasoning: Whether the query requires strong reasoning capabilities
        fast_response: Whether response speed is a priority
        cost_sensitive: Whether to prioritize lower-cost models
        ctx: MCP context object
    
    Returns:
        JSON object with the recommended model and details
    """
    try:
        # Get DEEPMYST_API_KEY from environment
        api_key = os.environ.get("DEEPMYST_API_KEY")
        if not api_key:
            error_msg = "DEEPMYST_API_KEY environment variable not set"
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
            
        ctx.info(f"Calling DeepMyst router API with payload: {payload}")
        
        # Call the router API
        async with aiohttp.ClientSession() as session:
            headers = {
                'Content-Type': 'application/json',
                'X-API-Key': api_key
            }
            
            try:
                async with session.post(ROUTER_URL, headers=headers, json=payload, timeout=50) as response:
                    if response.status == 200:
                        routing_data = await response.json()
                        ctx.info(f"Router API response: {routing_data}")
                        
                        # Parse and enhance the response
                        result = {
                            "recommended_model": routing_data.get("selected_llm", "gpt-4o"),
                            "provider": PROVIDER_MODEL_MAP.get(routing_data.get("selected_llm", "gpt-4o"), "unknown"),
                            "model_scores": routing_data.get("model_scores", {}),
                            "query_analysis": routing_data.get("query_analysis", {})
                        }
                        
                        return result
                    else:
                        error_text = await response.text()
                        error_msg = f"Router API error: {response.status} - {error_text}"
                        ctx.error(error_msg)
                        return {"error": error_msg, "fallback_model": "gpt-4o"}
            except Exception as e:
                error_msg = f"Network error when calling router API: {str(e)}"
                ctx.error(error_msg)
                return {"error": error_msg, "fallback_model": "gpt-4o"}
                
    except Exception as e:
        error_msg = f"Exception when calling router API: {str(e)}"
        ctx.error(error_msg)
        return {"error": error_msg, "fallback_model": "gpt-4o"}

@mcp.tool()
async def optimized_completion(
    prompt: str,
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
        model: Base model to use (will add -optimize flag)
        system_message: Optional system message
        temperature: Controls randomness (0-1)
        max_tokens: Maximum tokens in the response
        ctx: MCP context object
    
    Returns:
        The model's response text
    """
    try:
        client = get_client()
        
        # Add optimization flag
        optimized_model = f"{model}-optimize"
        
        # Prepare messages
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        messages.append({"role": "user", "content": prompt})
        
        # Call DeepMyst API
        ctx.info(f"Calling DeepMyst with optimized model: {optimized_model}")
        response = client.chat.completions.create(
            model=optimized_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
    except Exception as e:
        error_msg = f"Error in optimized_completion: {str(e)}"
        ctx.error(error_msg)
        return f"An error occurred: {str(e)}"

@mcp.tool()
async def auto_routed_completion(
    prompt: str,
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
        base_model: Base model to use (will add -auto flag)
        system_message: Optional system message
        temperature: Controls randomness (0-1)
        max_tokens: Maximum tokens in the response
        ctx: MCP context object
    
    Returns:
        The model's response text
    """
    try:
        client = get_client()
        
        # Add auto routing flag
        routed_model = f"{base_model}-auto"
        
        # Prepare messages
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        messages.append({"role": "user", "content": prompt})
        
        # Call DeepMyst API
        ctx.info(f"Calling DeepMyst with auto-routed model: {routed_model}")
        response = client.chat.completions.create(
            model=routed_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
    except Exception as e:
        error_msg = f"Error in auto_routed_completion: {str(e)}"
        ctx.error(error_msg)
        return f"An error occurred: {str(e)}"

@mcp.tool()
async def deepmyst_completion(
    prompt: str,
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
        client = get_client()
        
        # Build model name with flags
        model_name = base_model
        if auto_route:
            model_name += "-auto"
        if optimize:
            model_name += "-optimize"
        
        # Prepare messages
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        messages.append({"role": "user", "content": prompt})
        
        # Call DeepMyst API
        ctx.info(f"Calling DeepMyst with model: {model_name}")
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
    except Exception as e:
        error_msg = f"Error in deepmyst_completion: {str(e)}"
        ctx.error(error_msg)
        return f"An error occurred: {str(e)}"

@mcp.tool()
async def smart_completion(
    prompt: str,
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
        system_message: Optional system message
        optimize: Whether to enable token optimization
        temperature: Controls randomness (0-1)
        max_tokens: Maximum tokens in the response
        ctx: MCP context object
    
    Returns:
        The model's response text
    """
    try:
        client = get_client()
        
        # First, get the best model for this query using the router
        router_response = await get_best_model_for_query(prompt, ctx=ctx)
        
        # Check if the router call was successful
        if "error" in router_response and not "recommended_model" in router_response:
            # Use a default model if the router failed
            selected_model = "gpt-4o"
            ctx.warning(f"Router API call failed, using fallback model: {selected_model}")
        else:
            # Use the recommended model from the router
            selected_model = router_response.get("recommended_model", "gpt-4o")
            ctx.info(f"Using router-recommended model: {selected_model}")
        
        # Add optimization flag if requested
        if optimize:
            selected_model += "-optimize"
        
        # Prepare messages
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        messages.append({"role": "user", "content": prompt})
        
        # Call DeepMyst API
        ctx.info(f"Calling DeepMyst with model: {selected_model}")
        response = client.chat.completions.create(
            model=selected_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
    except Exception as e:
        error_msg = f"Error in smart_completion: {str(e)}"
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
      Use the tools provided by this MCP server to access these features.
    """

# API health check endpoint
@mcp.resource("deepmyst://health")
def get_health_check() -> str:
    """Get server health status"""
    try:
        # Check if API key is set
        api_key = os.environ.get("DEEPMYST_API_KEY")
        if not api_key:
            return json.dumps({
                "status": "error",
                "message": "DEEPMYST_API_KEY environment variable not set"
            })
            
        return json.dumps({
            "status": "healthy",
            "version": "1.0.0",
            "server": "deepmyst_mcp",
            "transport": "SSE and STDIO supported"
        })
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": str(e)
        })

# SSE transport handlers
async def handle_sse(scope, receive, send):
    """Handle SSE connection requests"""
    async with sse.connect_sse(scope, receive, send) as streams:
        logger.info("New SSE connection established")
        await mcp.run_with_transport(streams[0], streams[1])

async def handle_post(scope, receive, send):
    """Handle POST requests for client-to-server messages"""
    await sse.handle_post_message(scope, receive, send)

# Run the server
if __name__ == "__main__":
    # Check for environment variables
    api_key = os.environ.get("DEEPMYST_API_KEY")
    if not api_key:
        logger.error("DEEPMYST_API_KEY environment variable not set")
        print("Error: DEEPMYST_API_KEY environment variable not set")
        print("Please set your DeepMyst API key with:")
        print("  export DEEPMYST_API_KEY=your-api-key  # Linux/macOS")
        print("  set DEEPMYST_API_KEY=your-api-key     # Windows")
        sys.exit(1)
        
    # Check for command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--stdio":
        # Run with stdio transport (for Claude Desktop)
        logger.info("Starting DeepMyst MCP server with stdio transport")
        print("Starting DeepMyst MCP server with stdio transport")
        print("Press Ctrl+C to exit")
        mcp.run()
    else:
        # Run with SSE transport (for HTTP clients)
        port = int(os.environ.get("PORT", 8000))
        host = os.environ.get("HOST", "0.0.0.0")
        
        logger.info(f"Starting DeepMyst MCP server with SSE transport on {host}:{port}")
        print(f"Starting DeepMyst MCP server with SSE transport on {host}:{port}")
        print("Press Ctrl+C to exit")
        
        # Create the SSE transport
        sse = SseServerTransport("/mcp/message")
        
        # Configure Starlette app with routes
        app = Starlette(routes=[
            Route("/mcp/sse", endpoint=handle_sse),
            Route("/mcp/message", endpoint=handle_post, methods=["POST"]),
        ])
        
        # Run the server with uvicorn
        uvicorn.run(app, host=host, port=port)