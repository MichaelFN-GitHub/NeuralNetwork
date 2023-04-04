import numpy as np
import random as rand

class TicTacToe:
    def __init__(self):
        self.X = 0
        self.O = 1        
        self.winningPositions = [
            0b111000000, 0b000111000, 0b000000111,  # rows
            0b100100100, 0b010010010, 0b001001001,  # columns
            0b100010001, 0b001010100                # diagonals
        ]
        
        self.reset_board()
        
        
    def reset_board(self):
        self.state = np.array([0,0,0,0,0,0,0,0,0]).reshape(-1, 1)
        self.pieces = [0b0, 0b0]
        self.currentPlayer = self.X
        
        self.gameOver = False
        self.result = None
    
    
    
    def make_move(self, move):
        if self.gameOver:
            print("Game is already over.")
            return
        
        if not(self.is_legal_move(move)):
            print("Illegal move.")
            return
        
        position = 1 << move
        
        if self.currentPlayer == self.X:
            self.pieces[self.X] |= position
            self.state[move] = 1
        else:
            self.pieces[self.O] |= position
            self.state[move] = -1
        
        self.check_for_win()
        
        if not self.gameOver:
            self.currentPlayer = 1 - self.currentPlayer
            
            
    
    def undo_move(self, move):

        position = 1 << move
        
        if self.currentPlayer == self.X:
            self.pieces[self.X] &= ~position
        else:
            self.pieces[self.O] &= ~position
        self.state[move] = 0
        
        self.gameOver = False
        
        self.currentPlayer = 1 - self.currentPlayer
    
    
    
    def check_for_win(self):
        for position in self.winningPositions:
            if (self.currentPlayer == self.X and (self.pieces[self.X] & position) == position) or \
               (self.currentPlayer == self.O and (self.pieces[self.O] & position) == position):
                player = "X" if self.currentPlayer == self.X else "O"
                print(f"Player {player} wins!")
                self.gameOver = True
                self.result = 1 if self.currentPlayer == self.X else -1
                return
        
        if (self.pieces[self.X] | self.pieces[self.O]) == 0b111111111:
            print("It's a tie!")
            self.gameOver = True
            self.result = 0
          
            
          
    def is_legal_move(self, move):
        position = 1 << move
        return not(move >= 0 and move < 9 and (self.pieces[self.X] & position) or (self.pieces[self.O] & position))
       
    
    
    def print_board(self):
        symbols = ['_', 'X', 'O']  # empty, X, O
        rows = []
        for i in range(3):
            row = []
            for j in range(3):
                mask = 1 << (i * 3 + j)
                if self.pieces[self.X] & mask:
                    row.append(symbols[self.X+1])
                elif self.pieces[self.O] & mask:
                    row.append(symbols[self.O+1])
                else:
                    row.append(symbols[0])
            rows.append(row)
        
        print('+-----------+')
        for i in range(3):
            print('|', rows[i][0], '|', rows[i][1], '|', rows[i][2], '|')
            print('+-----------+')
        print()

        
        
    def get_state(self):
        return self.state
    
    
    def get_legal_moves(self):
        return self.state == 0


    def get_result(self):
        return self.result
    
    
    def make_ramdom_move(self):
        move = rand.randint(0,8)
        while not(self.is_legal_move(move)):
            move = rand.randint(0,8)
        self.make_move(move)
        
        
    

runGameLoop = False
    
# Game loop
if (runGameLoop):
    print("Game has started.")
    gameState = TicTacToe()
    gameState.print_board()
    while not(gameState.gameOver):
        userMove = int(input("Your move: "))
        gameState.make_move(userMove)
        gameState.print_board()
    
