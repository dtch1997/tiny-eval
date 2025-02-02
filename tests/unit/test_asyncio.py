import pytest
import asyncio
from tiny_eval.utils.asyncio import chain, batch_chain

# Helper async functions for testing
async def add_one(x):
    await asyncio.sleep(0.1)  # Simulate async work
    return x + 1

async def multiply_by_two(x):
    await asyncio.sleep(0.1)  # Simulate async work
    return x * 2

async def return_list(x):
    await asyncio.sleep(0.1)
    return [x, x + 1]

async def process_list(x):
    await asyncio.sleep(0.1)
    return [x * 2]

@pytest.mark.asyncio
async def test_chain_basic():
    # Test basic chaining of two functions
    composed = chain(add_one, multiply_by_two)
    result = await composed(5)
    assert result == 12  # (5 + 1) * 2

@pytest.mark.asyncio
async def test_chain_multiple():
    # Test chaining multiple functions
    composed = chain(add_one, chain(multiply_by_two, add_one))
    result = await composed(5)
    assert result == 13  # ((5 + 1) * 2) + 1

@pytest.mark.asyncio
async def test_batch_chain_basic():
    # Test basic batch chaining with lists
    composed = batch_chain(return_list, process_list)
    result = await composed(5)
    assert result == [10, 12]  # [5*2, 6*2]

@pytest.mark.asyncio
async def test_batch_chain_single_result():
    # Test batch chain when second function returns single values
    async def single_value(x):
        await asyncio.sleep(0.1)
        return x * 2

    composed = batch_chain(return_list, single_value)
    result = await composed(5)
    assert result == [10, 12]  # [5*2, 6*2]

@pytest.mark.asyncio
async def test_batch_chain_empty_list():
    # Test batch chain with empty list
    async def return_empty(_):
        await asyncio.sleep(0.1)
        return []

    composed = batch_chain(return_empty, process_list)
    result = await composed(5)
    assert result == []

@pytest.mark.asyncio
async def test_batch_chain_error_handling():
    # Test error handling
    async def raise_error(_):
        await asyncio.sleep(0.1)
        raise ValueError("Test error")

    composed = batch_chain(return_list, raise_error)
    with pytest.raises(ValueError):
        await composed(5)

@pytest.mark.asyncio
async def test_chain_multiple_funcs():
    # Test chaining three functions passed as arguments
    composed = chain(add_one, multiply_by_two, add_one)
    result = await composed(5)
    assert result == 13  # ((5 + 1) * 2) + 1

@pytest.mark.asyncio
async def test_chain_single_func():
    # Test chain with single function
    composed = chain(add_one)
    result = await composed(5)
    assert result == 6

@pytest.mark.asyncio
async def test_batch_chain_multiple_funcs():
    async def triple_list(x):
        await asyncio.sleep(0.1)
        return [x * 3]

    # Test chaining three functions that handle lists
    composed = batch_chain(return_list, process_list, triple_list)
    result = await composed(5)
    assert result == [30, 36]  # [((5*2)*3), ((6*2)*3)]

@pytest.mark.asyncio
async def test_batch_chain_mixed_returns():
    async def add_to_each(x):
        await asyncio.sleep(0.1)
        return x + 1  # Returns single value

    async def make_list(x):
        await asyncio.sleep(0.1)
        return [x, x + 10]  # Returns list

    # Test mixing functions that return lists and single values
    composed = batch_chain(return_list, add_to_each, make_list)
    result = await composed(5)
    assert result == [6, 16, 7, 17]  # [5->6->[6,16], 6->7->[7,17]]
