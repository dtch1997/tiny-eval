import pytest
from unittest.mock import AsyncMock, Mock

from steg_games.core.constants import Model
from steg_games.core.types import Message, Choice, Question
from steg_games.solve import build_solver

@pytest.fixture
def mock_model_api():
    api = AsyncMock()
    api.get_response = AsyncMock(return_value=[
        Choice(message=Message(role="assistant", content="Test response"))
    ])
    return api

@pytest.fixture
def mock_scaffold():
    scaffold = Mock()
    scaffold.format_prompt = Mock(return_value="Formatted prompt")
    scaffold.parse_response = Mock(return_value="Parsed answer")
    return scaffold

@pytest.mark.asyncio
async def test_build_solver_basic_flow(mock_model_api, mock_scaffold):
    # Arrange
    model = Model.GPT_4o
    solver = build_solver(model, mock_model_api, mock_scaffold)
    
    context = [Message(role="system", content="System message")]
    prompt = "Test prompt"
    question = Question(context=context, prompt=prompt)

    # Act
    results = await solver(question)

    # Assert
    assert len(results) == 1
    assert results[0] == "Parsed answer"
    
    # Verify scaffold interactions
    mock_scaffold.format_prompt.assert_called_once_with("Test prompt")
    mock_scaffold.parse_response.assert_called_once_with("Test response")
    
    # Verify model API interactions
    expected_messages = context + [Message(role="user", content="Formatted prompt")]
    mock_model_api.get_response.assert_called_once_with(expected_messages, model)