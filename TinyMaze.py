# TinyMaze.py
# A very simple maze game!
# Moves: "a" (left), "d" (right), "w" (up), "z" (down). Press Q to quit.

import sys
import random

class TinyMazeEnv():

	# define status codes
	stepped = 1
	blocked = 2
	won = 3
	quit = 4

	# define mazes
	mazes =	{ 
			  13: [ [ 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1],
			  		[ 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1],
			  		[ 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0],
			  		[ 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0],
			  		[ 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
			  		[ 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1],
			  		[ 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0],
			  		[ 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1],
			  		[ 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
			  		[ 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1],
			  		[ 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1],
			  		[ 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1],
			  		[ 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, -1] ],

			  11: [	[ 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0],
			 		[ 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1],
			 		[ 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
			 		[ 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1],
			 		[ 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1],
			 		[ 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0],
			 		[ 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
			 		[ 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0],
			 		[ 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1],
			 		[ 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
			 		[ 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, -1] ],

		      9:  [	[ 0, 0, 1, 0, 0, 0, 0, 1, 1],
			 		[ 1, 0, 0, 0, 1, 1, 0, 0, 1],
			 		[ 0, 0, 1, 0, 0, 0, 0, 0, 1],
			 		[ 1, 0, 1, 0, 1, 0, 1, 0, 0],
			 		[ 0, 0, 0, 1, 1, 0, 0, 0, 1],
			 		[ 0, 1, 0, 0, 0, 1, 0, 1, 1],
			 		[ 0, 0, 0, 1, 0, 0, 0, 0, 1],
			 		[ 1, 0, 0, 0, 0, 1, 1, 0, 1],
			 		[ 0, 0, 1, 1, 0, 0, 0, 0, -1] ],	

			  7:  [ [ 0, 0, 1, 0, 0, 0, 0],
			 		[ 1, 0, 0, 0, 1, 1, 0],
			 		[ 0, 0, 1, 0, 0, 0, 0],
			 		[ 1, 0, 1, 0, 1, 0, 1],
			 		[ 0, 0, 0, 1, 1, 0, 0],
			 		[ 0, 1, 0, 0, 0, 1, 0],
			 		[ 0, 0, 0, 1, 0, 0, -1] ],

			  5:  [	[ 0, 0, 1, 0, 1],
					[ 1, 0, 1, 0, 0],
					[ 0, 0, 0, 0, 1],
					[ 0, 1, 1, 0, 1],
					[ 0, 0, 1, 0, -1] ]
			}

	
	def __init__(self,maze_size=5):
		# initialize starting position and maze
		self.x = 0
		self.y = 0
		self.total_steps = 0
		if maze_size in self.mazes.keys():
			self.maze = self.mazes[maze_size]
		else:
			self.maze = self.mazes[5]
		self.maze_size = len(self.maze)

	def display_maze(self,move='None'):
		# display the maze in its current state
		print("\n" * 100) 	# clear screen
		offset = " " * int((self.maze_size-5) * 1.5)
		print(offset + "Welcome to TinyMaze!")
		print(offset + "        w: up")
		print(offset + "   a: left s: right")
		print(offset + "       z: down")
		print(offset + "Try to reach the X...")
		print("")
		print("---" * (self.maze_size + 2))
		for i in range(self.maze_size):
			row = " | "
			for j in range(self.maze_size):
				if i == self.y and j == self.x: 
					row += " U "
				elif self.maze[i][j] == 1: 
					row += "###"
				elif self.maze[i][j] == -1: 
					row += " X "
				else: 
					row += "   "
			print(row + " | ")
			print("---" * (self.maze_size + 2))
		print(offset + "Move: {0}  Total steps: {1}".format(move,self.total_steps))
		print(offset + "     x: {0}, y: {1}".format(self.x,self.y))

	def step(self,move):
		# process a single action
		self.total_steps += 1
		status = self.blocked
		if move == "a":
			if (self.x > 0) and (self.maze[self.y][self.x-1] != 1): 
				self.x -= 1
				status = self.stepped
		elif move == "s":
			if (self.x < self.maze_size-1) and (self.maze[self.y][self.x+1] != 1): 
				self.x += 1
				status = self.stepped
		elif move == "w":
			if (self.y > 0) and (self.maze[self.y-1][self.x] != 1): 
				self.y -= 1
				status = self.stepped
		elif move == "z":
			if (self.y < self.maze_size-1) and (self.maze[self.y+1][self.x] != 1): 
				self.y += 1
				status = self.stepped
		elif move == "Q":
			status = self.quit

		# check for a win
		if self.maze[self.y][self.x] == -1:
			status = self.won
		return status

	def play(self):
		self.display_maze()
		while True:
			if sys.version_info[0] < 3:
				move = raw_input()
			else:
				move = input()
			status = self.step(move)
			if status == self.won: 
				print("You won!")
				break
			elif status == self.quit: 
				print("You quit.")
				break
			else: 
				self.display_maze(move)

# main
if __name__ == "__main__":
	if len(sys.argv) > 1:
		maze = TinyMazeEnv(int(sys.argv[1]))
	else:
		maze = TinyMazeEnv()
	maze.play()
