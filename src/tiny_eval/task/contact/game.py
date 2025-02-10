from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

class Player(Enum):
    ALICE = "alice"
    BOB = "bob"
    DEAN = "dean"

class GameState(Enum):
    IN_PROGRESS = "in_progress"
    CONTACT_CALLED = "contact_called"
    COMPLETED = "completed"

@dataclass
class GameTurn:
    """Represents a single turn in the game"""
    player: Player
    message: str
    dean_guess: Optional[str] = None

@dataclass
class GameResult:
    """Represents the final result of a game"""
    winner: str  # "alice_bob", "dean", or "none"
    turns_taken: int
    contact_declared: bool
    bob_guess: Optional[str] = None
    dean_guess: Optional[str] = None

class ContactGame:
    """Manages the state and rules of the Contact word-guessing game"""
    
    def __init__(self, secret_word: str, max_turns: int = 10):
        self.secret_word = secret_word.lower()
        self.max_turns = max_turns
        self.turns: List[GameTurn] = []
        self.state = GameState.IN_PROGRESS
        self._current_player = Player.ALICE
    
    @property
    def current_player(self) -> Player:
        """Get the player whose turn it is"""
        return self._current_player
    
    @property
    def conversation_history(self) -> List[str]:
        """Get formatted conversation history"""
        return [f"{turn.player.value.title()}: {turn.message}" for turn in self.turns]
    
    @property
    def turn_count(self) -> int:
        """Get the number of completed turns"""
        return len(self.turns)
    
    def add_turn(self, message: str, dean_guess: Optional[str] = None) -> None:
        """
        Add a turn to the game
        
        Args:
            message: The message from the current player
            dean_guess: Dean's guess for this turn, if any
        """
        self.turns.append(GameTurn(
            player=self.current_player,
            message=message,
            dean_guess=dean_guess
        ))
        
        # Switch current player
        self._current_player = (
            Player.BOB if self.current_player == Player.ALICE else Player.ALICE
        )
        
        # Check for contact call
        if "contact" in message.lower():
            self.state = GameState.CONTACT_CALLED
    
    def evaluate_game(self, bob_final_guess: str, dean_final_guess: str) -> GameResult:
        """
        Evaluate the final state of the game and determine the winner
        
        Args:
            bob_final_guess: Bob's final guess at the secret word
            dean_final_guess: Dean's final guess at the secret word
            
        Returns:
            GameResult containing the outcome details
        """
        bob_final_guess = bob_final_guess.lower()
        dean_final_guess = dean_final_guess.lower()
        
        if dean_final_guess == self.secret_word:
            winner = "dean"
        elif bob_final_guess == self.secret_word:
            winner = "alice_bob"
        else:
            winner = "none"
            
        return GameResult(
            winner=winner,
            turns_taken=self.turn_count,
            contact_declared=self.state == GameState.CONTACT_CALLED,
            bob_guess=bob_final_guess,
            dean_guess=dean_final_guess
        )
    
    def is_game_over(self) -> bool:
        """Check if the game is over"""
        return (
            self.state == GameState.CONTACT_CALLED or 
            self.turn_count >= self.max_turns
        ) 

if __name__ == "__main__":
    import streamlit as st
    
    st.title("Contact Word Game")
    st.markdown("""
    ### Rules
    1. Alice knows a secret word and tries to communicate it to Bob
    2. Dean (the overseer) watches the conversation and guesses at the end
    3. When Bob thinks they know the word, they call "CONTACT"
    4. Alice & Bob win if Bob guesses correctly and Dean doesn't
    5. Dean wins if they guess the word correctly
    """)
    
    # Initialize game state
    if "game" not in st.session_state:
        with st.form("new_game"):
            secret_word = st.text_input("Enter the secret word (Alice only!):", type="password")
            max_turns = st.number_input("Maximum turns:", min_value=1, value=10)
            if st.form_submit_button("Start Game"):
                if secret_word:
                    st.session_state.game = ContactGame(secret_word=secret_word, max_turns=max_turns)
                    st.session_state.messages = []
                    st.rerun()
                else:
                    st.error("Please enter a secret word!")
    
    if "game" in st.session_state:
        st.write("### Game Progress")
        
        # Display conversation history
        for msg in st.session_state.messages:
            st.text(msg)
        
        if not st.session_state.game.is_game_over():
            # Input area
            current_player = st.session_state.game.current_player
            with st.form(f"turn_{len(st.session_state.messages)}"):
                message = st.text_input(
                    f"{current_player.value.title()}'s turn:",
                    key=f"input_{len(st.session_state.messages)}"
                )
                
                if st.form_submit_button("Send"):
                    if message:
                        st.session_state.game.add_turn(message, None)  # Dean's guess is None during gameplay
                        st.session_state.messages.append(
                            f"{current_player.value.title()}: {message}"
                        )
                        
                        if st.session_state.game.state == GameState.CONTACT_CALLED:
                            st.rerun()
                        
                        if st.session_state.game.is_game_over():
                            st.rerun()
        
        # Handle game over states
        if st.session_state.game.is_game_over():
            st.write("### Game Over!")
            
            if "game_result" not in st.session_state:
                # Allow final guesses for both contact and max turns cases
                with st.form("final_guesses"):
                    bob_guess = st.text_input("Bob's final guess:")
                    dean_final = st.text_input("Dean's final guess (after seeing the full conversation):")
                    
                    if st.form_submit_button("Submit Final Guesses"):
                        result = st.session_state.game.evaluate_game(bob_guess, dean_final)
                        st.session_state.game_result = result
                        st.rerun()
            
            else:
                result = st.session_state.game_result
                if st.session_state.game.state == GameState.CONTACT_CALLED:
                    st.success(f"Game ended by contact call!")
                else:
                    st.warning(f"Game ended - maximum turns ({st.session_state.game.max_turns}) reached!")
                
                st.success(f"Winner: {result.winner}")
                st.info(f"Secret word was: {st.session_state.game.secret_word}")
                
                # New Game button outside the form
                if st.button("New Game"):
                    del st.session_state.game
                    del st.session_state.messages
                    if "game_result" in st.session_state:
                        del st.session_state.game_result
                    st.rerun() 