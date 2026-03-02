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
        self.last_pos = None
        
    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
        
    
        
     
 
        
           
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
            
                
      
 
        


    

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
      
        actions = game_state.get_legal_actions(self.index)
   

            
      
        # You can profile your evaluation time by uncommenting these lines
        #start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
       # print ('eval time for agent %d: %.4f' % (self.index, time.time() - start))

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]
        
    

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
        
        

        
      #look-ahead strategy instead of deep-search
        best_val = float('-inf')
        best_actions = []
        for action in actions:
            succ1 = self.get_successor(game_state, action)
            val1 = self.evaluate(game_state, action)
            
            #look 1step further
            actions2 = succ1.get_legal_actions(self.index)
            if actions2:
                vals2 = {a2: self.evaluate(succ1, a2) for a2 in actions2}
                best_a2 = max(vals2, key=vals2.get)
                val2 = vals2[best_a2]
                #"3rd step look"
                succ2 = self.get_successor(succ1, best_a2)
                actions3 = succ2.get_legal_actions(self.index)
                if actions3:
                    val3 = max(self.evaluate(succ2, a3) for a3 in actions3)
                else:
                    val3 = 0
            else:
                val2 = 0
                val3 = 0
            total = val1 + 0.5 * val2 + 0.25 * val3 
            if total > best_val:
                best_val = total
                best_actions = [action]
            elif total == best_val:
                best_actions.append(action)
                
      
            
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
        # If we're in a tight corridor AND ghost is close, heavily penalize
        if num_exits <= 1:
            features['dead_end'] = 1
        elif num_exits == 2:
            features['dead_end'] = 0  # corridor but not a dead end
        else:
            features['dead_end'] = 0
            
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
        #features = self.get_features(game_state, action)

    
        w = {
        'successor_score': 100,
        'distance_to_food': -4,
        'distance_to_non_scared_ghosts': 18,
        'dead_end': -50,
        'distance_to_home': 0,
        'stop': - 500,
        'reverse': -30,
         'on_home_side': 0.01,
        'distance_to_power_capsule': -10
       
       
        }
               
         
        if carrying >= 5:
            w['distance_to_home'] = -30
            w['distance_to_food'] = 0
            
         # end
        if game_state.data.timeleft < 150 and carrying > 0:
            w['distance_to_home'] = -500
            w['distance_to_food'] = 0
            
        return w
    
   





class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """
  

     # register the initial state and all the free positions on own side
    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        self.patrol_index = 0 #add this

        #choose 3 points on own side of the grid, those are the 'patroeuillepunten'
        # agent will walk through these points when there is no invader
        
        if self.legal_home_positions:   #self.legal_home_positions is a list of the free points, given in ReflexCaptureAgent
            sorted_pos = sorted(self.legal_home_positions, key=lambda p: p[1])  # sort these on y-coördinate, from low y-value to high y-value
            self.patrol_points = [
                sorted_pos[0],
                sorted_pos[len(sorted_pos)//2],
                sorted_pos[-1]
            ] 
        #choose the one on posistion 0 (the lowest point), the one in the middle of this list (point somewhere in the middle), the one at the end of this list (highest point


    # look at the state after every action, the features are based on the future state, in that way the agent can predict the effect of his actions
    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # the agent is a defender = should stay on own side
        # bonus if agent is still on own side after an action
        #is the agent on our own side? (yes = 1, no = 0)

        features['on_defense'] = 1 if not my_state.is_pacman else 0


        # invaders 
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)

        if len(invaders) > 0:
            # distance to nearest invader
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)
        else:
            #no invaders: go to patrouillepunt
            target = self.patrol_points[self.patrol_index]
            features['patrol_distance'] = self.get_maze_distance(my_pos, target)
        
        #agent wants to minimise the distance, will move to patroeillepoint

        # defend own fooddots
        # food_defend_distence; the closer we are to our own food dots, the better we can defend them
        # food_defend_count; the less food there is left, the 
        """food_defend = self.get_food_you_are_defending(successor).as_list()
        if food_defend:
            min_food_dist = min(self.get_maze_distance(my_pos, f) for f in food_defend)
            features['food_defend_distance'] = min_food_dist
            features['food_defend_count'] = len(food_defend)

        
        # defend power capsule
        capsules_defend = self.get_capsules_you_are_defending(successor)
        if capsules_defend:
            min_capsule_dist = min(self.get_maze_distance(my_pos, c) for c in capsules_defend)
            features['capsule_defend_distance'] = min_capsule_dist"""
            features['capsule_defend_distance'] = min_capsule_dist


        # overall score; the higher the score the better: 
        features['successor_score'] = self.get_score(successor)

        #when agent is scared -> flee 
        """"NOG AAN TE PASSEN"""
        #features['scared'] = 1 if my_state.scared_timer > 0:
    


        #standing still or going back and forth (given in baseline. alr)    
        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1
        return features
    

    def get_weights(self, game_state, action):
        return {
            'on_defense': 100,
            'num_invaders': -1000,
            'invader_distance': -10,
            'patrol_distance': -2,
            'food_defend_distance': -5,
            'food_defend_count': 100,
            'capsule_defend_distance': -8,
            'successor_score': 100,
            'stop': -100,
            'reverse': -10,
            'scared': 0
        }
        
    #choose the action with the highest score, if there are no invaders -> go to nearest patroeiepunt
    def choose_action(self, game_state):
        action = super().choose_action(game_state)
        invaders = [a for a in self.get_opponents(game_state) if game_state.get_agent_state(a).is_pacman]
        if len(invaders) == 0:
            self.patrol_index = (self.patrol_index + 1) % len(self.patrol_points)
        return action
    # when there is an invader; patroeille stops, invader_distance -feature with a strong negative wheight, so he wants to go there)