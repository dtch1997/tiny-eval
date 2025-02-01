import asyncio
from typing import Callable, Awaitable, List, Any, Union

def to_coros(items: List[Any]) -> List[Awaitable[Any]]:
    """Convert a list of items into a list of coroutines that return those items.
    
    Args:
        items: List of items to convert to coroutines
        
    Returns:
        List of coroutines that return the corresponding items when awaited
    """
    async def make_coro(x: Any) -> Any:
        return x
        
    return [make_coro(item) for item in items]


def chain(
    *funcs: Callable[[Any], Awaitable[Any]]
) -> Callable[[Any], Awaitable[Any]]:
    """Chain multiple async functions together, passing the output of each as input to the next.
    
    Args:
        *funcs: Variable number of async functions to chain together. Each function takes
               the output of the previous function as input.
        
    Returns:
        An async function that chains all functions together
    """
    async def composed(x: Any) -> Any:
        result = x
        for func in funcs:
            result = await func(result)
        return result
    return composed

def batch_chain(
    *funcs: Callable[[Any], Awaitable[Union[List[Any], Any]]]
) -> Callable[[Any], Awaitable[List[Any]]]:
    """Chain multiple async functions where each can return a list or single value.
    Each function's results are flattened and passed to the next function.
    
    Args:
        *funcs: Variable number of async functions to chain together. Each function
               can return either a list or single value which will be flattened.
        
    Returns:
        An async function that chains all functions together, flattening list results
    """
    async def composed(x: Any) -> List[Any]:
        current_results = [x]
        
        for func in funcs:
            # Create coroutines for each current result
            coroutines = [func(y) for y in current_results]
            
            # Await all coroutines
            results = await asyncio.gather(*coroutines)
            
            # Flatten results - if any are lists, unpack them
            current_results = []
            for result in results:
                if isinstance(result, list):
                    current_results.extend(result)
                else:
                    current_results.append(result)
                    
        return current_results
    
    return composed