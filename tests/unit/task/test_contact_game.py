import pytest
from tiny_eval.task.contact.game import ContactGame, Player, GameState, GameResult

def test_game_initialization():
    """Test game initialization with default values"""
    game = ContactGame("test")
    assert game.secret_word == "test"
    assert game.max_turns == 10
    assert game.turn_count == 0
    assert game.current_player == Player.ALICE
    assert game.state == GameState.IN_PROGRESS

def test_add_turn():
    """Test adding turns to the game"""
    game = ContactGame("test")
    
    # First turn - Alice
    game.add_turn("Hello", "no guess")
    assert game.turn_count == 1
    assert game.current_player == Player.BOB
    assert game.conversation_history == ["Alice: Hello"]
    
    # Second turn - Bob
    game.add_turn("Hi there", "still no guess")
    assert game.turn_count == 2
    assert game.current_player == Player.ALICE
    assert len(game.conversation_history) == 2
    assert game.conversation_history[-1] == "Bob: Hi there"

def test_contact_detection():
    """Test that the game detects when contact is called"""
    game = ContactGame("test")
    
    game.add_turn("Hello", "no guess")
    assert game.state == GameState.IN_PROGRESS
    
    game.add_turn("CONTACT!", "maybe test?")
    assert game.state == GameState.CONTACT_CALLED

def test_game_over_conditions():
    """Test different game over conditions"""
    game = ContactGame("test", max_turns=2)
    
    # Not over after first turn
    game.add_turn("Hello", "no guess")
    assert not game.is_game_over()
    
    # Over after max turns
    game.add_turn("Hi there", "no guess")
    assert game.is_game_over()
    
    # Over immediately when contact called
    game = ContactGame("test")
    game.add_turn("CONTACT!", "guess")
    assert game.is_game_over()

def test_evaluate_game():
    """Test game evaluation with different outcomes"""
    game = ContactGame("test")
    
    # Dean wins
    result = game.evaluate_game("wrong", "test")
    assert result.winner == "dean"
    assert result.turns_taken == 0
    assert not result.contact_declared
    
    # Alice/Bob win
    result = game.evaluate_game("test", "wrong")
    assert result.winner == "alice_bob"
    
    # Nobody wins
    result = game.evaluate_game("wrong", "also_wrong")
    assert result.winner == "none"

def test_case_insensitive_evaluation():
    """Test that word matching is case insensitive"""
    game = ContactGame("Test")
    
    result = game.evaluate_game("TEST", "test")
    assert result.winner == "dean"  # Dean guessed correctly despite case difference

def test_conversation_history_formatting():
    """Test that conversation history is properly formatted"""
    game = ContactGame("test")
    
    game.add_turn("Hello there", "no guess")
    game.add_turn("Hi back", "still no guess")
    
    history = game.conversation_history
    assert history == [
        "Alice: Hello there",
        "Bob: Hi back"
    ]

def test_max_turns_zero():
    """Test game behavior with zero max turns"""
    game = ContactGame("test", max_turns=0)
    assert game.is_game_over()

def test_empty_message():
    """Test handling of empty messages"""
    game = ContactGame("test")
    game.add_turn("", "no guess")
    assert game.conversation_history == ["Alice: "] 