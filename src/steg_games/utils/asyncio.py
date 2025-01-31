import asyncio
from typing import TypeVar, Callable, Awaitable, List, Any, Union

T = TypeVar('T')
U = TypeVar('U') 
V = TypeVar('V')

def chain(
    f: Callable[[T], Awaitable[U]], 
    g: Callable[[U], Awaitable[V]]
) -> Callable[[T], Awaitable[V]]:
    """Chain two async functions together, passing the output of f as input to g.
    
    Args:
        f: First async function to execute
        g: Second async function to execute with f's result
        
    Returns:
        An async function that chains f and g together
    """
    async def composed(x: T) -> V:
        return await g(await f(x))
    return composed

def batch_chain(
    f: Callable[[T], Awaitable[List[U]]], 
    g: Callable[[U], Awaitable[Union[V, List[V]]]]
) -> Callable[[T], Awaitable[List[V]]]:
    """Chain two async functions where f returns a list and g is applied to each element.
    
    Args:
        f: First async function that returns a list
        g: Second async function to apply to each element of f's result
        
    Returns:
        An async function that chains f and g together, flattening list results from g
    """
    async def composed(x: T) -> List[V]:
        # Get list of intermediate results from f
        intermediate_results = await f(x)
        
        # Create coroutines for each intermediate result
        coroutines = [g(y) for y in intermediate_results]
        
        # Await all coroutines and flatten results
        results = await asyncio.gather(*coroutines)
        return [z for sublist in results if isinstance(sublist, list) for z in sublist] or results
    
    return composed