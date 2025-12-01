# Team strategy for Pacman Capture the Flag.
# Focus on aggressive scared ghost hunting, strategic defense, and mode stability.

import random
import math
import time

import contest.util as util
from contest.capture_agents import CaptureAgent
from contest.game import Directions


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

        # AJUSTES DE CAUTELA/AGRESIVIDAD
        self.return_threshold = 6
        self.search_timeout = 0.35
        self.danger_radius = 3
        self.capsule_radius = 8
        
        # NUEVOS CONTADORES PARA HISTERESIS (PERSISTENCIA DE MODO)
        self.evade_timer = 0
        self.min_evade_steps = 15 # Mínimo de pasos para permanecer en Evade

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
        
        # Fantasmas no asustados y cercanos (PELIGRO)
        chasing_ghosts = [
            g for g in visible_ghosts 
            if g.scared_timer == 0 and self.get_maze_distance(my_pos, g.get_position()) <= self.danger_radius
        ]
        
        # Fantasma asustado (OPORTUNIDAD)
        scared_targets = [
            g for g in visible_ghosts 
            if g.scared_timer > 0 
        ]

        # ----------------------------------------------------
        # Decide behaviour mode (con Histeresis)
        # ----------------------------------------------------
        mode = "collect"
        target = self.start # Objetivo de fallback

        if self.evade_timer > 0:
            # 1. ESTAMOS EN MODO EVASIÓN FORZADA (PERSISTENCIA)
            self.evade_timer -= 1
            mode = "evade"
            target = self.closest_home(game_state, my_pos)
            
            # Si el agente llega a su lado, la evasión forzada termina
            if not state.is_pacman:
                 self.evade_timer = 0
        
        if chasing_ghosts:
            # 2. PELIGRO INMINENTE: INICIAR EVASIÓN FORZADA
            mode = "evade"
            target = self.closest_home(game_state, my_pos)
            # Reiniciar el contador de evasión
            if self.evade_timer == 0:
                 self.evade_timer = self.min_evade_steps
                 
        elif scared_targets:
            # 3. CAZAR FANTASMAS ASUSTADOS
            target_ghost = min(scared_targets, key=lambda g: self.get_maze_distance(my_pos, g.get_position()))
            # Si el timer está muy bajo y llevamos comida, volvemos a casa.
            if target_ghost.scared_timer < 5 and state.num_carrying > 0:
                mode = "return"
                target = self.closest_home(game_state, my_pos)
            else:
                mode = "hunt_scared"
                target = target_ghost.get_position()

        elif state.num_carrying >= self.return_threshold:
            # 4. RETORNO PARA ANOTAR PUNTOS.
            mode = "return"
            target = self.closest_home(game_state, my_pos)
        
        else:
            # 5. RECOLECCIÓN Y ATAQUE ESTRATÉGICO.
            mode = "collect"
            food = self.get_food(game_state).as_list()
            capsules = self.get_capsules(game_state)
            
            target = self.best_offensive_target(game_state, my_pos, food, capsules, visible_ghosts)

        # A* → next step only
        nxt = self.astar_next(game_state, my_pos, target, mode)
        if not nxt:
            return random.choice(legal)

        return self.direction_from(my_pos, nxt)

    # -----------------------------------------------
    # RESTO DE FUNCIONES (best_offensive_target, closest_home, astar_next, step_cost, direction_from)
    # Se mantienen IGUALES a la versión V2 (cuidado con copiar solo la parte superior).
    # -----------------------------------------------

    def best_offensive_target(self, game_state, pos, food_list, capsules, ghosts):
        """
        Elige entre Power Capsules cercanas y la comida más segura/cercana.
        """
        # Prioridad a las cápsulas cercanas
        near_capsules = [
            c for c in capsules 
            if self.get_maze_distance(pos, c) <= self.capsule_radius
        ]
        if near_capsules:
            return min(near_capsules, key=lambda c: self.get_maze_distance(pos, c))

        if not food_list:
            return self.start

        # Comida que no está demasiado cerca de fantasmas no asustados
        safe_food = []
        for f in food_list:
            danger = False
            for g in ghosts:
                # Solo fantasmas NO asustados son una amenaza aquí
                if g.scared_timer == 0 and self.get_maze_distance(f, g.get_position()) <= self.danger_radius:
                    danger = True
                    break
            if not danger:
                safe_food.append(f)

        if safe_food:
            # Elige la comida segura más cercana
            return min(safe_food, key=lambda f: self.get_maze_distance(pos, f))
        else:
            # Si toda la comida está cerca de fantasmas, elige la comida más cercana asumiendo riesgo
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
        Costo personalizado que penaliza fantasmas no asustados y premia la caza.
        """
        base = 1.0

        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        ghosts = [e for e in enemies if not e.is_pacman and e.get_position()]

        for g in ghosts:
            g_pos = g.get_position()
            d = self.get_maze_distance(pos, g_pos)

            if g.scared_timer > 0:
                # Fantasma Asustado: REDUCCIÓN AGRESIVA DEL COSTO para acercarse.
                if mode == "hunt_scared":
                    base -= (40 - g.scared_timer) * 0.5 
                    if d <= 1: base = 0.01
                # En otros modos, se ignora la penalización.
            else:
                # Fantasma NO Asustado: Aumento del costo (Amenaza).
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
#                 DEFENSIVE AGENT            #
##############################################

# NOTA: EL AGENTE DEFENSIVO NO NECESITA HISTERESIS EN ESTA VERSIÓN.
# Mantiene el código de la versión V2.

class HybridDefensive(CaptureAgent):

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        self.distancer.get_maze_distances()
        self.start = game_state.get_agent_position(self.index)

        self.patrol_points = self.compute_patrol(game_state)
        self.current_patrol_index = 0
        self.last_seen_invader = None

    # -----------------------------------------------

    def choose_action(self, game_state):
        legal = game_state.get_legal_actions(self.index)
        legal = [a for a in legal if a != Directions.STOP]
        if not legal:
            return Directions.STOP

        my_pos = game_state.get_agent_position(self.index)

        invaders = self.get_invaders(game_state)

        if invaders:
            inv = min(invaders, key=lambda s: self.get_maze_distance(my_pos, s.get_position()))
            self.last_seen_invader = inv.get_position() 
            target = self.last_seen_invader
            mode = "chase"
        
        elif self.last_seen_invader:
            if self.get_maze_distance(my_pos, self.last_seen_invader) < 5:
                target = self.last_seen_invader
                mode = "chase_last_seen"
            else:
                self.last_seen_invader = None
                target = self.patrol_points[self.current_patrol_index]
                mode = "patrol"
        
        else:
            target = self.patrol_points[self.current_patrol_index]
            mode = "patrol"
            
            if my_pos == target:
                self.current_patrol_index = (self.current_patrol_index + 1) % len(self.patrol_points)
                target = self.patrol_points[self.current_patrol_index]
        
        nxt = self.astar_next(game_state, my_pos, target, mode)
        if not nxt:
            return random.choice(legal)

        return self.direction_from(my_pos, nxt)

    # -----------------------------------------------

    def compute_patrol(self, game_state):
        walls = game_state.get_walls()
        food_on_side = self.get_food_you_are_defending(game_state).as_list()
        
        if food_on_side:
            food_on_side.sort(key=lambda p: manhattan(self.start, p), reverse=True)
            return food_on_side[:min(6, len(food_on_side))]
        
        mid = walls.width // 2
        x_home = mid - 1 if self.red else mid
        candidates = [(x_home, y) for y in range(walls.height) if not walls[x_home][y]]
        
        return random.sample(candidates, min(6, len(candidates))) if candidates else [self.start]


    # -----------------------------------------------

    def get_invaders(self, game_state):
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        return [e for e in enemies if e.is_pacman and e.get_position()]

    def astar_next(self, game_state, start, goal, mode):
        walls = game_state.get_walls()
        start = tuple(start)
        goal = tuple(goal)

        frontier = util.PriorityQueue()
        frontier.push(start, 0)

        came = {}
        cost = {start: 0}

        t0 = time.time()
        search_timeout = 0.35

        while not frontier.is_empty():
            if time.time() - t0 > search_timeout:
                return None

            cur = frontier.pop()
            if cur == goal:
                break

            for nxt in neighbours(cur, walls):
                c = cost[cur] + 1
                if nxt not in cost or c < cost[nxt]:
                    cost[nxt] = c
                    frontier.push(nxt, c + manhattan(nxt, goal))
                    came[nxt] = cur

        if goal not in came:
            return None

        step = goal
        while came[step] != start:
            step = came[step]
        return step

    def direction_from(self, a, b):
        ax, ay = a
        bx, by = b
        if bx == ax + 1: return Directions.EAST
        if bx == ax - 1: return Directions.WEST
        if by == ay + 1: return Directions.NORTH
        if by == ay - 1: return Directions.SOUTH
        return Directions.STOP