import numpy as np

class TicTacToe:
    """
        Board is represented by 18 bit, where
            LSBs 0-8 are the placement of Xs.
            LSBs 9-17 are the placement of Os.
            
        The following is a mapping of the 18 LSBs to a board:
            Black bits:     White bits:
            100000000       001010000
            
            Board:
            X - O
            - X -
            - - -
    """
    
    PLAYER_WHITE = 0
    PLAYER_BLACK = 9
    
    WHITE_WIN = 1
    BLACK_WIN = -1
    DRAW = 0
    
    
    # Constructor
    def __init__(self):
        self.restart_game()
        
    
    # Retarts the game
    def restart_game(self):
        self.board = 0                     #Init empty board
        self.player = self.PLAYER_WHITE    #Init first player
        self.legal_moves = [0,1,2,3,4,5,6,7,8]
        self.game_over = False
        self.one_hot_state = np.zeros((9,1))
        self.one_hot_input = 1
        
        
    # Get one hot state
    def get_state(self):
        return self.one_hot_state
        
        
    # Get legal moves
    def get_legal_moves(self):
        return self.legal_moves
        
    
    # Makes move by current player on the given square
    def make_move(self, move):
        
        # Return if game is already over
        if (self.game_over):
            return False
        
        """
        Player black is set to 9, so that the
        move can be easily shifted to the right bit:
            If white to move, then (square 3) << player = 000000000 000001000
            if black to move, then (square 3) << player = 000001000 000000000
        """
        # Move the move on the board
        self.board |= (1 << (move + self.player))
        
        # Update legal moves
        self.legal_moves.remove(move)
        
        # Update one hot state
        self.one_hot_state[move] = self.one_hot_input

        # Check for win
        self.game_over = self.check_for_win()
        
        # Change player
        self.change_player()
        
        
        return True
    
    
    # Check if game is over
    def check_for_win(self):
        win_combinations = [
            0b111000000, 0b000111000, 0b000000111,  # Rows
            0b100100100, 0b010010010, 0b001001001,  # Columns
            0b100010001, 0b001010100                # Diagonals
            ]
        
        # Get bit board for each player
        white_board = self.board & 0b111111111
        black_board = (self.board >> self.PLAYER_BLACK) & 0b111111111
        
        # Check every win combination and return winner if found
        for combination in win_combinations:
            if ((white_board & combination) == combination):
                self.winner = self.WHITE_WIN
                return True
            elif ((black_board & combination) == combination):
                self.winner = self.BLACK_WIN
                return True
        
        # Draw if no win comibnation was found and board is full
        if ((white_board | black_board) == 0b111111111):
            self.winner = self.DRAW
            return True
        
        # No combination was found and board not full -> game is not over
        return False
        
        
    # Changes player
    def change_player(self):
        self.player = self.PLAYER_BLACK - self.player
        self.one_hot_input = -self.one_hot_input
        
        
    # Print the board
    def __str__(self):
        s = "\n"
        
        # Print winner if game over, else prints player to move
        if (self.game_over):
            s += "Game over."
        
            
        # Print the board
        for i in range(9):
            
            # New line every 3 squares
            if (i%3 == 0):
                s += "\n"
              
            # Add "X" if white, "O" if black, "-" if empty
            if ((self.board >> (self.PLAYER_WHITE + i)) & 1 == 1):
                s += "X"
            elif ((self.board >> (self.PLAYER_BLACK + i)) & 1 == 1):
                s += "O"
            else:
                s += "-"
                
        return s


# %% Test game and print board
import random

game = TicTacToe()

for i in range(9):
    legal_moves = game.get_legal_moves()
    if not game.game_over:
        random.shuffle(legal_moves)
        game.make_move(legal_moves[0])
    print(game)
print(game.get_state())


# %% Check time to play 1 million games
import time
import random

game = TicTacToe()

start_time = time.time()
num_games = 1000000
for i in range(num_games):
    
    if (i%100000 == 0):
        elapsed_time = time.time() - start_time
        print(f"{i} games played in {elapsed_time} seconds")
    
    game.restart_game()
    for i in range(9):
        legal_moves = game.get_legal_moves()
        if not game.game_over:
            random.shuffle(legal_moves)
            game.make_move(legal_moves[0])

elapsed_time = time.time() - start_time
print(f"Time to play 1 million games: {elapsed_time} seconds")

# %% Test
import random

game = TicTacToe()
legal_moves = game.get_legal_moves()
random.shuffle(legal_moves)