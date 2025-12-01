# Team strategy for Pacman Capture the Flag.
# Focus on aggressive scared ghost hunting, strategic defense, and mode stability.

import random
import math
import time
import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from game import Actions



#######################
# Team Initialization #
#######################

def create_team(first_index, second_index, is_red,
                first_agent='HybridOffensive', second_agent='HybridDefensive'):
    """
    Creates your two agents.
    """
    return [eval(first_agent)(first_index), eval(second_agent)(second_index)]


##########################
# COMMON HELPER ROUTINES #
##########################

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def neighbours(pos, walls):
    """Returns legal neighbouring positions."""
    x, y = pos
    res = []
    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < walls.width and 0 <= ny < walls.height:
            if not walls[nx][ny]:
                res.append((nx,ny))
    return res


##############################################
#         OFFENSIVE / HYBRID AGENT           #
##############################################

class HybridOffensive(CaptureAgent):

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        self.distancer.get_maze_distances()
        self.start = game_state.get_agent_position(self.index)

        # Fixed tunning Values
        self.return_threshold = 6
        self.search_timeout = 0.35
        self.danger_radius = 3
        self.capsule_radius = 8
        self.evade_timer = 0
        self.min_evade_steps = 15 
    # -----------------------------------------------

    def choose_action(self, game_state):
        legal = game_state.get_legal_actions(self.index)
        legal = [a for a in legal if a != Directions.STOP]
        if not legal:
            return Directions.STOP

        my_pos = game_state.get_agent_position(self.index)
        state = game_state.get_agent_state(self.index)

        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        visible_ghosts = [e for e in enemies if not e.is_pacman and e.get_position()]
        
        # Near ghosts, not scared
        chasing_ghosts = [
            g for g in visible_ghosts 
            if g.scared_timer == 0 and self.get_maze_distance(my_pos, g.get_position()) <= self.danger_radius
        ]
        
        # Ghost scared
        scared_targets = [
            g for g in visible_ghosts 
            if g.scared_timer > 0 
        ]

        # ----------------------------------------------------
        # Decide behaviour mode 
        # ----------------------------------------------------
        mode = "collect"
        target = self.start 

        # Evade mode
        if self.evade_timer > 0:
            self.evade_timer -= 1
            mode = "evade"
            target = self.closest_home(game_state, my_pos)
            
            # If pac-man reached null evade
            if not state.is_pacman:
                 self.evade_timer = 0
        
        # Evade ghost
        if chasing_ghosts:
            mode = "evade"
            target = self.closest_home(game_state, my_pos)

            # Reset
            if self.evade_timer == 0:
                 self.evade_timer = self.min_evade_steps
        
        # Chase ghost
        elif scared_targets:
            target_ghost = min(scared_targets, key=lambda g: self.get_maze_distance(my_pos, g.get_position()))
            if target_ghost.scared_timer < 5 and state.num_carrying > 0:
                mode = "return"
                target = self.closest_home(game_state, my_pos)
            else:
                mode = "hunt_scared"
                target = target_ghost.get_position()

        # Back home to note points
        elif state.num_carrying >= self.return_threshold:
            mode = "return"
            target = self.closest_home(game_state, my_pos)
        
        # Take food
        else:
            mode = "collect"
            food = self.get_food(game_state).as_list()
            capsules = self.get_capsules(game_state)
            
            target = self.best_offensive_target(game_state, my_pos, food, capsules, visible_ghosts)

        # A* for next step
        nxt = self.astar_next(game_state, my_pos, target, mode)
        if not nxt:
            return random.choice(legal)

        return self.direction_from(my_pos, nxt)

    # -----------------------------------------------
    # Other functions used for OFFENSIVE
    # -----------------------------------------------

    def best_offensive_target(self, game_state, pos, food_list, capsules, ghosts):
        """
        Chosse near safest food
        """
        # Prioroity to near capusles
        near_capsules = [
            c for c in capsules 
            if self.get_maze_distance(pos, c) <= self.capsule_radius
        ]
        if near_capsules:
            return min(near_capsules, key=lambda c: self.get_maze_distance(pos, c))

        if not food_list:
            return self.start

        # Food away from ghosts
        safe_food = []
        for f in food_list:
            danger = False
            for g in ghosts:

                # Not scared Ghosts
                if g.scared_timer == 0 and self.get_maze_distance(f, g.get_position()) <= self.danger_radius:
                    danger = True
                    break
            if not danger:
                safe_food.append(f)

        if safe_food:
            # Choose food near to yourself
            return min(safe_food, key=lambda f: self.get_maze_distance(pos, f))
        else:
            # Assume risk in case no other option is aviable
            return min(food_list, key=lambda f: self.get_maze_distance(pos, f))

    # -----------------------------------------------

    def closest_home(self, game_state, pos):
        walls = game_state.get_walls()
        mid = walls.width // 2
        x_home = mid - 1 if self.red else mid
        candidates = [(x_home, y) for y in range(walls.height) if not walls[x_home][y]]
        if not candidates: return self.start
        return min(candidates, key=lambda p: self.get_maze_distance(pos, p))

    # -----------------------------------------------

    def astar_next(self, game_state, start, goal, mode):
        """Simplified A*: returns only the next step towards goal."""
        walls = game_state.get_walls()
        start = tuple(start)
        goal = tuple(goal)

        frontier = util.PriorityQueue()
        frontier.push(start, 0)

        came = {}
        cost = {start: 0}

        start_time = time.time()
        search_timeout = self.search_timeout

        while not frontier.is_empty():
            if time.time() - start_time > search_timeout:
                return None

            cur = frontier.pop()
            if cur == goal:
                break

            for nxt in neighbours(cur, walls):
                new_cost = cost[cur] + self.step_cost(game_state, nxt, mode)
                if nxt not in cost or new_cost < cost[nxt]:
                    cost[nxt] = new_cost
                    pr = new_cost + manhattan(nxt, goal)
                    frontier.push(nxt, pr)
                    came[nxt] = cur

        if goal not in came:
            return None

        step = goal
        while came[step] != start:
            step = came[step]
        return step

    # -----------------------------------------------

    def step_cost(self, game_state, pos, mode):
        """
        Penalized cost when offensive
        """
        
        base = 1.0

        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        ghosts = [e for e in enemies if not e.is_pacman and e.get_position()]

        for g in ghosts:
            g_pos = g.get_position()
            d = self.get_maze_distance(pos, g_pos)

            # Chase scared ghost
            if g.scared_timer > 0:
                if mode == "hunt_scared":
                    base -= (40 - g.scared_timer) * 0.5 
                    if d <= 1: base = 0.01
            
            # Evade normal ghost depending on proximity
            else:
                if d == 0:
                    return 9999
                elif d == 1:
                    base += 150 
                elif d == 2:
                    base += 30
                elif d <= 4:
                    base += 5

        if mode == "return" or mode == "evade":
            base *= 0.8
        
        return max(0.01, base)

    # -----------------------------------------------

    def direction_from(self, a, b):
        ax, ay = a
        bx, by = b
        if bx == ax + 1: return Directions.EAST
        if bx == ax - 1: return Directions.WEST
        if by == ay + 1: return Directions.NORTH
        if by == ay - 1: return Directions.SOUTH
        return Directions.STOP


##############################################
#              DEFENSIVE AGENT               #
##############################################

class HybridDefensive(CaptureAgent):

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        self.distancer.get_maze_distances()
        self.start = game_state.get_agent_position(self.index)

        self.patrol_points = self.compute_patrol(game_state)
        self.current_patrol_index = 0

        self.last_seen_invader = None
        self.prev_position = None    
        self.last_action = None

    # -------------------------------------------------------------

    def choose_action(self, game_state):
        legal = [a for a in game_state.get_legal_actions(self.index)
                 if a != Directions.STOP]

        if not legal:
            return Directions.STOP

        my_pos = game_state.get_agent_position(self.index)
        invaders = self.get_invaders(game_state)

        # Chase invader
        if invaders:
            inv = min(invaders, key=lambda s: self.get_maze_distance(my_pos, s.get_position()))
            self.last_seen_invader = inv.get_position()
            target = self.last_seen_invader

        # Chase last seen
        elif self.last_seen_invader:
            if self.get_maze_distance(my_pos, self.last_seen_invader) <= 6:
                target = self.last_seen_invader
            else:
                self.last_seen_invader = None
                target = self.patrol_points[self.current_patrol_index]

        # Patrol Mode
        else:
            target = self.patrol_points[self.current_patrol_index]
            if my_pos == target:
                self.current_patrol_index = (self.current_patrol_index + 1) % len(self.patrol_points)
                target = self.patrol_points[self.current_patrol_index]

        # Next Step
        nxt = self.astar_next(game_state, my_pos, target)

        if not nxt:
            chosen = random.choice(legal)
        else:
            chosen = self.direction_from(my_pos, nxt)

        # Aviod osiclation
        next_pos = Actions.get_successor(my_pos, chosen)

        if self.prev_position == next_pos:  
            options = [a for a in legal if a != chosen]
            if options:
                chosen = random.choice(options)

        self.prev_position = my_pos
        self.last_action = chosen

        return chosen

    # -----------------------------------------------
    # Other functions used for DEFENSIVE
    # -----------------------------------------------

    def compute_patrol(self, game_state):
        food = self.get_food_you_are_defending(game_state).as_list()
        walls = game_state.get_walls()

        if food:
            food.sort(key=lambda p: manhattan(self.start, p), reverse=True)
            return food[:min(6, len(food))]

        mid = walls.width // 2
        home_x = mid - 1 if self.red else mid
        candidates = [(home_x, y) for y in range(walls.height) if not walls[home_x][y]]

        return random.sample(candidates, min(6, len(candidates))) if candidates else [self.start]

    # -------------------------------------------------------------

    def get_invaders(self, game_state):
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        return [e for e in enemies if e.is_pacman and e.get_position()]

    # -------------------------------------------------------------

    def astar_next(self, game_state, start, goal):
        walls = game_state.get_walls()
        start = tuple(start)
        goal = tuple(goal)

        frontier = util.PriorityQueue()
        frontier.push((start, None), 0)

        came = {}
        cost = {start: 0}

        t0 = time.time()

        while not frontier.is_empty():
            if time.time() - t0 > 0.35:
                return None

            (cur, action_to_cur) = frontier.pop()

            if cur == goal:
                break

            for nxt, act in self.neighbours_with_actions(cur, walls):
                reverse_penalty = 4 if self.last_action and act == Directions.REVERSE[self.last_action] else 0
                new_cost = cost[cur] + 1 + reverse_penalty

                if nxt not in cost or new_cost < cost[nxt]:
                    cost[nxt] = new_cost
                    came[nxt] = cur
                    priority = new_cost + manhattan(nxt, goal)
                    frontier.push((nxt, act), priority)

        if goal not in came:
            return None

        step = goal
        while came[step] != start:
            step = came[step]

        return step

    # -------------------------------------------------------------

    def neighbours_with_actions(self, pos, walls):
        x, y = pos
        dirs = {
            Directions.NORTH: (x, y + 1),
            Directions.SOUTH: (x, y - 1),
            Directions.EAST:  (x + 1, y),
            Directions.WEST:  (x - 1, y)
        }
        return [(p, d) for d, p in dirs.items() if not walls[p[0]][p[1]]]

    # -------------------------------------------------------------

    def direction_from(self, a, b):
        ax, ay = a
        bx, by = b
        if bx == ax + 1: return Directions.EAST
        if bx == ax - 1: return Directions.WEST
        if by == ay + 1: return Directions.NORTH
        if by == ay - 1: return Directions.SOUTH
        return Directions.STOP
