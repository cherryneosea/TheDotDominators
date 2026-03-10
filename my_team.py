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
        
    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
        
        #modification 2:  precompute all dead ends
        
     
 
        
           
        grid_width = game_state.get_walls().width
        grid_height = game_state.get_walls().height
        mid_grid = grid_width // 2
        #home positions for my team
        self.legal_home_positions = []

        #determining our team
        if self.red:
            #red is on left-side of board
            home_x = mid_grid - 1
        else: #blue team
            home_x = mid_grid 

        for i in range(grid_height):
            if not game_state.has_wall(home_x, i):
                self.legal_home_positions.append((home_x, i))
            
                
    def minimax(self, game_state, depth, agent_index, ghost_index, alpha=float('-inf'), beta=float('inf')):
        #edge case
        if depth == 0:
            return self.evaluate(game_state, Directions.STOP) 
        
        actions = game_state.get_legal_actions(agent_index)
        
        #pacman turn = max
        if agent_index == self.index:
            best = float('-inf')
            for action in actions:
                successor = self.get_successor(game_state, action)
                val = self.minimax(successor, depth - 1, ghost_index, ghost_index, alpha, beta)
                best = max(best, val)
                alpha = max(alpha, best)
                if beta <= alpha: # prune
                    break
            return best
        
        else: #ghost layer = min
            
            best = float('inf')
            for action in actions:
                successor = game_state.generate_successor(agent_index, action) 
                val = self.minimax(successor, depth, self.index, ghost_index)
                best = min(best, val)
                beta = min(beta, best)
                if beta <= alpha: #prune
                    break
            return best
        
 
        


    ##disquqlified, timeout , or more than 3secs computations times

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
      
        actions = game_state.get_legal_actions(self.index)
   

            
      
        # You can profile your evaluation time by uncommenting these lines
        start = time.time()
        #values = [self.evaluate(game_state, a) for a in actions]
       #print ('eval time for agent %d: %.4f' % (self.index, time.time() - start))

        #max_value = max(values)
       # best_actions = [a for a, v in zip(actions, values) if v == max_value]
        
    

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
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
        
        
        #mini minimax-lookahead strategy 
        #modification 1: alpha-beta pruning to improve optimality, cause wth was that
        """we need the visible ghosts"""
        enemies = self.get_opponents(game_state)
        visible = []
        for i in enemies:
            enemy_state = game_state.get_agent_state(i)
            enemy_pos = enemy_state.get_position()
            #when position in none = ghost not visible
            if not enemy_state.is_pacman and enemy_pos is not None:
                my_pos = game_state.get_agent_state(self.index).get_position()
                d = self.get_maze_distance(my_pos, enemy_pos)
                if d <= 5:
                    visible.append(i)
        
        
        if visible:
            ghost_index = visible[0]
            values = [self.minimax(self.get_successor(game_state, action), 1, ghost_index, ghost_index) for action in actions]
        else:
            values = [self.evaluate(game_state, a) for a in actions]
        max_val = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_val]
        
        #print ('eval time for agent %d: %.4f' % (self.index, time.time() - start))


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
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """


    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        
        
        #because get food returns a grid-obj 2D boolean array
        food_list = self.get_food(successor).as_list()
        #power-capsules list
        power_capsules = self.get_capsules(successor)
        #power_capsules_lst = power_capsules.as_list()
        #scared and non-scared ghosts of the opponent
        #opponents = self.get_opponents(successor).as_list()
        enemies = self.get_opponents(successor)

        #immediate gain after eating food
        features['successor_score'] = -len(food_list)  # self.get_score(successor)
        """how many foods are left to track progress"""
      
        features['num_capsules_left'] =len(power_capsules)

        #my implemented features!

        #compute distance to the nearest power-capsule=> important so larger weight
        
        """if len(power_capsules) > 0:
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance_to_power_capsule = min(
                [self.get_maze_distance(my_pos, capsule) for capsule in  power_capsules]
            )
            features['distance_to_power_capsule'] = min_distance_to_power_capsule"""
            #return features  
            
        #min distance to scared ghost => good since when eating a capsule they become scared 
        scared_ghosts = []
        non_scared_ghosts = []
        #separate opponenets and store into lists=> enemy is just an idx
        for enemy in enemies:
            agent_state = successor.get_agent_state(enemy)
            is_scared = agent_state.scared_timer > 0
            ghost_pos = agent_state.get_position()
            # ghost agents may be invisible
            if not agent_state.is_pacman and ghost_pos is not None:
                if is_scared:
                    scared_ghosts.append(ghost_pos)
                else:
                    non_scared_ghosts.append(ghost_pos)
             

        #compute min distance to different ghosts
        #edible ghosts....
        """if scared_ghosts != []:
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance_to_scared = min(
                [self.get_maze_distance(my_pos, scared) for scared in scared_ghosts]
            )
            features['distance_to_scared_ghosts'] = min_distance_to_scared
        else: 
            #not visible => so they are not a factor in our calculations
            features['distance_to_scared_ghosts'] = 100"""

        #dangerous ghosts 
        if len(non_scared_ghosts) > 0:
            my_pos = successor.get_agent_state(self.index).get_position()
            
            min_distance_to_notscared = min(
                [self.get_maze_distance(my_pos, not_scared) for not_scared in non_scared_ghosts]
            )
            features['distance_to_non_scared_ghosts'] = min_distance_to_notscared
        else:
            #no visible non scared ghosts-> default value
            features['distance_to_non_scared_ghosts'] = 999

        features['distance_to_scared_ghosts'] = 100  # just hardcode this


        # Compute distance to the nearest 3 food dots
        agent_state = successor.get_agent_state(self.index)
        my_pos = agent_state.get_position()
        if food_list:
            # filter first using manhattan
            food_distances = [(util.manhattan_distance(my_pos, f), f) for f in food_list]
            food_distances.sort()
            top_foods = [f for _, f in food_distances[:3]]
            min_food_dist = min([self.get_maze_distance(my_pos, f) for f in top_foods])
            features['distance_to_food'] = min_food_dist
       
        #distance to home for pacman to get home when he is in danger and has food
  

        distance_to_home = min([util.manhattan_distance(my_pos, home_position) for home_position in self.legal_home_positions])
        features['distance_to_home'] = distance_to_home
        

        if action == Directions.STOP:
            features['stop'] = 1
        else:
            features['stop'] = 0

        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1
       
           
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
                features['dead_end'] = 3  # very dangerous
            elif ghost_dist < 10:
                features['dead_end'] = 1  # risky
            else:
                features['dead_end'] = 0  # safe, ghost is far
        else:
            features['dead_end'] = 0
       # ghost_dist = features.get('distance_to_non_scared_ghosts', 999)
        # If we're in a tight corridor AND ghost is close, heavily penalize
        '''if num_exits <= 1:
            features['dead_end'] = 1
        elif num_exits == 2:
            features['dead_end'] = 0  # corridor but not a dead end
        else:
            features['dead_end'] = 0'''
            
        # penalty when he should be hunting instead of staying on his side
        if not agent_state.is_pacman:
            features['on_home_side'] = 1
        else:
            features['on_home_side'] = 0
            
        # progress bias
        current_direction = game_state.get_agent_state(self.index).configuration.direction
        if action == current_direction:
            features['progress'] = 1  
        elif action == Directions.REVERSE[current_direction]:
            features['reverse'] = 1  
            




    
        return features
    



#there is currently way too many features and edge cases, which makes pacman overthink and be coward
#minimize features and few distinct modes and every possible case
# add new weights for each extra feature you add => making it context dependant
    def get_weights(self, game_state, action):
      
        carrying = game_state.get_agent_state(self.index).num_carrying
        features = self.get_features(game_state, action)
        ghost_d = features.get( 'distance_to_non_scared_ghosts', 999)
        
    

    
        w = {
        'successor_score': 100,
        'distance_to_food': -5,
        'distance_to_non_scared_ghosts': 15,
        'dead_end': -80,
        'distance_to_home': 0,
        'stop': - 500,
        'reverse': -80,
         'on_home_side': -20,
        'distance_to_power_capsule': -10,
        
       
       
        }
        
               
         
        if carrying >= 3:
            w['distance_to_home'] = -20
            w['distance_to_food'] = 0
            
        '''if ghost_d >= 6:
            w['distance_to_food'] = -20
            w['successor_score'] = 200'''

            
            
         # end
        if game_state.data.timeleft < 150 and carrying > 0:
            w['distance_to_home'] = -500
            w['distance_to_food'] = 0
            
            
        return w
            
    
   





class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    defends own side of the grid
    works in 3 cases: 1: invader is visible (so within 5 steps): follows invader in order to catch him
                      2: invader on own side but not visible: move with noisy distance towards it
                      3: no opponent on own side: pattrouilleer through 5 fixed points, near the border 

                      extra: in case agent is scared = opponent took capsule, agent flees from invaders.

    """

     # register the initial state and all the free positions on own side
    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        

        #choose 5 points on own side of the grid, those are the 'patroeuillepunten'
        # agent will walk through these points when there is no invader
        
        if self.legal_home_positions:   #self.legal_home_positions is a list of the free points, given in ReflexCaptureAgent
            sorted_pos = sorted(self.legal_home_positions, key=lambda p: p[1])
            n = len(sorted_pos)
            # sort these on y-coördinate, from low y-value to high y-value
            self.patrol_points = [
                sorted_pos[0],
                sorted_pos[n // 4],
                sorted_pos[n // 2],
                sorted_pos[3 * n // 4],
                sorted_pos[-1],
            ]

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
            'on_defense': 100,
            'num_invaders': -1000,
            'invader_distance': -10,
            'patrol_distance': -5,
            'successor_score': 100,
            'stop': -100,
            'reverse': -10,
        }

    
    def choose_action(self, game_state):
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
    