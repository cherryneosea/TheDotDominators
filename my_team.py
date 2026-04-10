# baseline_team.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baseline_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import util
import time

from capture_agents import CaptureAgent
from game import Directions
from util import nearest_point


#################
# Team creation # 
################# 



"""keep working """
def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None
        self.legal_home_positions = []
        self.patrol_poitns = []
        self.last_pos = None
        
    def compute_legal_home_positions(self, game_state):
        """this function precomputes all legal home positions"""
        grid_width = game_state.get_walls().width
        grid_height = game_state.get_walls().height
        mid_grid = grid_width // 2
        legal_home_positions = []
        #determining our team
        if self.red:
            #red is on left-side of board
            home_x = mid_grid - 1
        else: #blue team
            home_x = mid_grid 

        for i in range(grid_height):
            if not game_state.has_wall(home_x, i):
                legal_home_positions.append((home_x, i))
                
        return legal_home_positions
            
        
    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
        #added legal home positions at start of the game, since they are constant
        self.legal_home_positions = self.compute_legal_home_positions(game_state)
            
 
        



    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        #base greedy actions
        actions = game_state.get_legal_actions(self.index)
        values = [self.evaluate(game_state, a) for a in actions]
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]
        return random.choice(best_actions)
            
      

    

     
             

        
      
        
        
  

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        score = features*weights  
       
        return score

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
    This offensive agent uses minimax with alpha-beta pruning as an offense strategy when choosing 
    an action when ghosts are visible and greedy-search if not.
    The implemented features are meant to encourage pacman to collect as many food-dots with caution
    
    """

    #helper methods:
    def detect_dead_ends(self, game_state, successor, features):
            
        # Count available exits from current position (dead end detection)
        my_pos = successor.get_agent_state(self.index).get_position()
        walls = game_state.get_walls()
        x, y = int(my_pos[0]), int(my_pos[1])
        num_exits = sum([
             not walls[x+1][y],
            not walls[x-1][y],
             not walls[x][y+1],
             not walls[x][y-1]
            ])
        ghost_dist = features.get('distance_to_non_scared_ghosts', 999)
        if num_exits <= 1:
            if ghost_dist < 6:
                return 3  # very dangerous
            elif ghost_dist < 10:
                return 1  # risky
            else:
                return  0  # safe, ghost is far
        else:
            return 0
      
        
    def get_reverse_action(self, game_state, action):
        """returns whether the current action..."""
        reverse = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == reverse:
            return 1
        return 0
      
        
    def get_stop_action(self, action):
        """returns whether the agents stops"""
        if action == Directions.STOP:
            return 1
        else:
            return 0
        
    def distance_to_home(self, successor):
        """returns the ditance from current position of pacman to his side
        the legal_home_position are already precomuted when registering initial state"""
        my_pos = my_pos = successor.get_agent_state(self.index).get_position()
        distance_to_home = min([util.manhattan_distance(my_pos, home_position) for home_position in self.legal_home_positions])
        return distance_to_home
        
      
    def distance_to_dangerous_ghosts(self, successor):
        """returns the shortest distance between pacman and non_scared ghosts
            by using the get_maze_distance and not an estimate!"""
                
        non_scared_ghosts = []
            #separate opponenets and store into lists=> enemy is just an idx
        enemies = self.get_opponents(successor)
        for enemy in enemies:
            agent_state = successor.get_agent_state(enemy)
            is_scared = agent_state.scared_timer > 0
            ghost_pos = agent_state.get_position()
            #check whether ghosts are invisible
            if not agent_state.is_pacman and ghost_pos is not None:
                if not is_scared:
                    non_scared_ghosts.append(ghost_pos)
                 
            if len(non_scared_ghosts) > 0:
                my_pos = successor.get_agent_state(self.index).get_position()
                min_distance_to_notscared = min(
                [self.get_maze_distance(my_pos, not_scared) for not_scared in non_scared_ghosts]
            )
                return min_distance_to_notscared
            else:
            #no visible non scared ghosts-> default value
                return 999
            
                  
    def get_successor_score(self, food_lst):
        """return the immediate gain after eating a food dot"""
        return -len(food_lst)
        
    def distance_to_food_dots(self, successor, food_lst):
        """return the distance to the top 3 closest food dots by filtering
        them using manhattan_distance and then getting the exact distance in the maze
        to avoid computational overhead"""
        
        agent_state = successor.get_agent_state(self.index)
        my_pos = agent_state.get_position()
        if food_lst:
            # filter first using manhattan
            food_distances = [(util.manhattan_distance(my_pos, f), f) for f in food_lst]
            food_distances.sort()
            top_foods = [f for _, f in food_distances[:3]]
            min_food_dist = min([self.get_maze_distance(my_pos, f) for f in top_foods])
            return min_food_dist
        return 999
        
        
        
        

    def get_features(self, game_state, action):
        """The features implemented are:
            1. successor_score = immidiate gain after eating a food-dot
            2. distance_to_food = distance to the 3 closest food-dots
            3. distance_to_non_scared_ghosts = the minimum distance between pacman and a ghost
            4. distance_to_home = to indicate how far pacman is from his territory
            5. stop = to indicate whether pacman is moving or not
            6. reverse = to indicare whether pacman takes the action that's opposite of the current one
            7. dead_end = to indicate how many exits are available from current position"""
        
        features = util.Counter()
        #compute once for efficiency
        successor = self.get_successor(game_state, action)
        
        #because get food returns a grid-obj 2D boolean array
        food_list = self.get_food(successor).as_list()
      
        features["distance_to_food"] = self.distance_to_food_dots(successor, food_list)
        features["distance_to_non_scared_ghosts"] = self.distance_to_dangerous_ghosts(successor)
        features["successor_score"] = self.get_successor_score(food_list)
        features['distance_to_home'] = self.distance_to_home(successor)
        features["stop"] = self.get_stop_action(action)
        features["reverse"] = self.get_reverse_action(game_state, action)
        features["dead_end"] = self.detect_dead_ends(game_state, successor, features)
            
        return features
    




    def get_weights(self, game_state, action):
        """ weight-assignments for features:
            1. successor_score = gets a large positive weight, since eating collecting as many food-dots as possible is the goal!
            2. distance_to_food = relatively small negative weight, enough to make it attractive with caution 
            3. distance_to_non_scared_ghosts = large positive weight to indicate danger
            4. dead_end = large negative weight, to discourage pacman from getting there if unnecessary
            5. distance_to_home = an initial weight of 0, because pacman should collect food-dots 
            6. stop = pacman should never stopping moving or freeze in place and keeps moving, hence the heavy penalty
            7. reverse = to penalize oscillation, it gets a large negative weight
            """
      
        carrying = game_state.get_agent_state(self.index).num_carrying
        

        w = {
    'successor_score': 100,
    'distance_to_food': -10,
    'distance_to_non_scared_ghosts': 25,  
    'dead_end': -150,  
    'distance_to_home': 0,
    'stop': -500,
    'reverse': -80,
   
}
        
       #encouraging pacman to return with food-dots, rather than letting his greed overpower him if possible
       #to avoid suicidal behavior
        if carrying >= 6:
            w['distance_to_home'] = -30
            w['distance_to_food'] = 0
            
       
              
        # pacman should return home if time is almost up and is carrying some food
        if game_state.data.timeleft < 150 and carrying > 0:
            w['distance_to_home'] = -500
            w['distance_to_food'] = 0
            
            
        return w
    
    
    
    def minimax_alpha_beta(self, game_state, depth, agent_index, ghost_index, alpha=float('-inf'), beta=float('inf')):
        """minimax with alpha-beta pruning, for action evaluation"""
        #edge case
        if depth == 0:
            return self.evaluate(game_state, Directions.STOP) 
        
        actions = game_state.get_legal_actions(agent_index)
        
        #pacman's layer = max
        if agent_index == self.index:
            best = float('-inf')
            for action in actions:
                successor = self.get_successor(game_state, action)
                val = self.minimax_alpha_beta(successor, depth - 1, ghost_index, ghost_index, alpha, beta)
                best = max(best, val)
                alpha = max(alpha, best)
                if beta <= alpha: # prune
                    break
            return best
        
        else: #ghost's layer= min   
            best = float('inf')
            for action in actions:
                successor = game_state.generate_successor(agent_index, action) 
                val = self.minimax_alpha_beta(successor, depth, self.index, ghost_index)
                best = min(best, val)
                beta = min(beta, best)
                if beta <= alpha: #prune
                    break
            return best
       
    
    def no_food_left_return_home(self, game_state, actions, num_food_left): 
        """the purpose of this helper function is to return the best action when there is no food to collect"""
        if num_food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action
        
      
    def getting_visible_enemies(self, game_state):
        """this function returns visible opponents using partial observability property, 
        so pacman can use minimax to evaluate his actions """
        enemies = self.get_opponents(game_state)
        visible_e = []
        for i in enemies:
            enemy_state = game_state.get_agent_state(i)
            enemy_position = enemy_state.get_position()
            #when position in none = ghost not visible
            if not enemy_state.is_pacman and enemy_position is not None:
                my_pos = game_state.get_agent_state(self.index).get_position()
                d = self.get_maze_distance(my_pos, enemy_position)
                if d <= 5:
                    visible_e.append(i)
        return visible_e
           
          
    def choose_action(self, game_state):
        """this function returns the best action best using an 
        - adversarial search strategy: minimax with alpha beta pruning at depth 1 when ghosts are visible
        - greedy-strategy when the ghosts are invisible"""
        
        
        actions = game_state.get_legal_actions(self.index)
        food_left = len(self.get_food(game_state).as_list())
        
        #mission accomplished!, go home
        result = self.no_food_left_return_home(game_state, actions, food_left)
        if result is not None:
            return result


    
        visible_enemies = self.getting_visible_enemies(game_state)
        if visible_enemies:
            ghost_index = visible_enemies[0]
            values = [self.minimax_alpha_beta(self.get_successor(game_state, action), 1, ghost_index, ghost_index) for action in actions]
        else:
            values = [self.evaluate(game_state, a) for a in actions]
        max_val = max(values)
        
        best_actions = [a for a, v in zip(actions, values) if v == max_val]


        return random.choice(best_actions)
        
   





class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    defends own side of the grid
    works in 3 cases: 1: invader is visible (so within 5 steps): follows invader in order to catch him
                      2: invader on own side but not visible: move with noisy distance towards it
                      3: no opponent on own side: pattrouilleer through 5 fixed points, near the border, where only accesible point are used, and stays 5 boxes away from the upper and lower border

                      extra: in case agent is scared = opponent took capsule, agent flees from invaders.

    """

     # register the initial state and all the free positions on own side
    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        
        # define which border positions are accesible starting from the start point
        start = self.start
        reachable = []
        for pos in self.legal_home_positions:
            dist = self.get_maze_distance(start, pos)
            if dist < 10000:
                reachable.append(pos)
        # only allow y-coordinates that are between (height - 5) and 5
        grid_height = game_state.get_walls().height
        min_y_allowed = 5
        max_y_allowed = grid_height -5
        reachable = [p for p in reachable if min_y_allowed <= p[1] <= max_y_allowed]
        
        if not reachable: #no point in that zone then we go to the startposition
            self.patrol_points = [start]
        else: #sort on y-coordinate and choose 5 points, spread across the height of the grid
            reachable.sort(key=lambda p: p[1])
            n = len(reachable)
            if n <= 5:
                self.patrol_points = reachable
            else:
                indices = [0, n//4, n//2, 3*n//4, n-1]
                self.patrol_points = [reachable [i] for i in indices]

        self.patrol_index = 0
    

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # the agent is a defender = should stay on own side
        # bonus if agent is still on own side after an action
        #is the agent on our own side? (yes = 1, no = 0)

        features['on_defense'] = 1 if not my_state.is_pacman else 0

        # (EXTRA CASE) when opponent eat a capsule, flee away as long as scared_timer > 0

        if my_state.scared_timer > 0: 
            enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
            invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
            if invaders:
                dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
                #positive weight in get_weights -> bigger distance = higher score = flee
                features['scared_distance'] = min(dists)
            if action == Directions.STOP:
                features['stop'] = 1
            return features 


        # get the opponents which are on our side and which are visible (position known, within 5 steps)
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)

        if len(invaders) > 0:
            # (CASE 1) invader visible
            # exact position is known --> whats the maze distance and follow direclty
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)
        else:
            # (CASE 2) invader not visible but on own side
            #get_agents_distances() gives vor each agent the estimaded distance (with noise +- 6)
            # if the opponent is_pacman is --> he is on our side of the grid (although we dont see him or know his exact position)
            #agents moves in the direction where estimated distance becomes smaller
            noisy_distances = successor.get_agent_distances()
            opponent_indices = self.get_opponents(successor)
            enemy_estimates = [
                noisy_distances[i] for i in opponent_indices
                if successor.get_agent_state(i).is_pacman
            ]
            if enemy_estimates:
                features['invader_distance'] = min(enemy_estimates)
            else: 
                # (CASE 3) no invader on our side; patrouilleren, go to the next pattroeillepoint
                #patrol-index increases when arrived to ponit (choose_action)
                target = self.patrol_points[self.patrol_index % len(self.patrol_points)]
                features['patrol_distance'] = self.get_maze_distance(my_pos, target)
        
        
        features['successor_score'] = self.get_score(successor)


        if action == Directions.STOP: 
            features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: 
            features['reverse'] = 1

        return features
    

    def get_weights(self, game_state, action):
        my_state = game_state.get_agent_state(self.index)

        #different weights for scared case
        if my_state.scared_timer > 0:
            return {
                'on_defense':      200,
                'scared_distance':  50,
                'stop':           -200,
            }
        return {
            'on_defense': 1000,
            'num_invaders': -1000,
            'invader_distance': -50,
            'patrol_distance': -15,
            'successor_score': 100,
            'stop': -100,
            'reverse': -10,
        }

    
    def choose_action(self, game_state):
        #print('Mijn code is nu aan het runnen: defensive')
        action = super().choose_action(game_state)
        #check whether there is a visible invader
        invaders = [a for a in self.get_opponents(game_state)
                    if game_state.get_agent_state(a).is_pacman]


        # higher the patrol index when arrived at the current patrolpoint (distance <= 1).

        if not invaders:
            my_pos = game_state.get_agent_state(self.index).get_position()
            target = self.patrol_points[self.patrol_index % len(self.patrol_points)]
            if self.get_maze_distance(my_pos, target) <= 1:
                self.patrol_index = (self.patrol_index + 1) % len(self.patrol_points)

        return action
    