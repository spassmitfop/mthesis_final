from .game_objects import GameObject, NoObject, ValueObject
from ._helper_methods import _convert_number
import sys
import numpy as np

"""
RAM extraction for the game GALAXIAN. Supported modes: ram.
"""
# TODO: Enemy Ships could be up to 42, but that does not happen in an unmodified game > which one should be saved?
MAX_NB_OBJECTS = {'Player': 1, 'PlayerMissile': 1, 'EnemyShip': 42, 'DivingEnemy': 10, 'EnemyMissile': 39}
MAX_NB_OBJECTS_HUD = MAX_NB_OBJECTS | {'Life': 3, 'Score': 1, 'Round': 1}

# enemy_missiles saves the enemy missiles according to at which ram position their x positon is saved
enemy_missiles = [NoObject()] * MAX_NB_OBJECTS["EnemyMissile"]


class Player(GameObject):
    """
    The player figure i.e, the gun.
    """

    def __init__(self):
        super().__init__()
        self._xy = 66, 186
        self.wh = 8, 13
        self.rgb = 236, 236, 236
        self.hud = False


class PlayerMissile(GameObject):
    """
    The projectiles fired by the player.
    """

    def __init__(self):
        super().__init__()
        self._xy = 66, 186
        self.wh = 1, 3
        self.rgb = 210, 164, 74
        self.hud = False


class EnemyMissile(GameObject):
    """
    The projectiles fired by the Enemy.
    """

    def __init__(self):
        super().__init__()
        self._xy = 66, 186
        self.wh = 1, 4
        # self.rgb = 228, 111, 111
        self.rgb = 210, 164, 74
        self.hud = False
        self.y_ram_index = -1


class EnemyShip(GameObject):
    """
    The Enemy Ships.
    """

    def __init__(self):
        super().__init__()
        self._xy = 66, 186
        self.wh = 6, 9
        self.rgb = 232, 204, 99
        self.hud = False


class DivingEnemy(GameObject):
    """
    The Enemy which are currently attacing.
    """

    def __init__(self):
        super().__init__()
        self._xy = 66, 186
        self.wh = 8, 11
        self.rgb = 232, 204, 99
        self.hud = False


class Score(ValueObject):
    """
    The player's remaining lives (HUD).
    """

    def __init__(self):
        super().__init__()
        self.rgb = 232, 204, 99
        self._xy = 63, 4
        self.wh = 39, 7
        self.hud = True


class Round(ValueObject):
    """
    The round counter display (HUD).
    """

    def __init__(self):
        super().__init__()
        self.rgb = 214, 214, 214
        self._xy = 137, 188
        self.wh = 7, 7
        self.hud = True


class Life(GameObject):
    """
    The remaining lives of the player (HUD).
    """

    def __init__(self):
        super().__init__()
        self.rgb = 214, 214, 214
        self._xy = 19, 188
        self.wh = 3, 7
        self.hud = True


# parses MAX_NB* dicts, returns default init list of objects
def _get_max_objects(hud=False):
    def fromdict(max_obj_dict):
        objects = []
        mod = sys.modules[__name__]
        for k, v in max_obj_dict.items():
            for _ in range(0, v):
                objects.append(getattr(mod, k)())
        return objects

    if hud:
        return fromdict(MAX_NB_OBJECTS_HUD)
    return fromdict(MAX_NB_OBJECTS)


# TODO Update to fit current state
def _init_objects_ram(hud=False):
    """
    (Re)Initialize the objects
    """
    objects: list[GameObject] = [Player(), PlayerMissile()] + [NoObject()] * 93

    if hud:
        objects.extend([NoObject(), NoObject(), NoObject(), Score(), Round()])

    return objects


prev_ram = None

def _detect_objects_ram(objects, ram_state, hud=False):
    """
       For all objects:
       (x, y, w, h, r, g, b)
    """
    global prev_ram
    if prev_ram is None:
        prev_ram = ram_state

    player = objects[0]
    if ram_state[11] != 255:  # else player not on screen
        if type(player) == NoObject:
            player = Player()
            objects[0] = player
        player.x, player.y = ram_state[100] + 8, 170
    elif type(player) == Player:
        objects[0] = NoObject()

    player_missile = objects[1]
    if ram_state[11] != 255 and ram_state[11] != 151:  # else there is no missile
        if type(player_missile) == NoObject:
            player_missile = PlayerMissile()
            objects[1] = player_missile
        player_missile.x, player_missile.y = ram_state[60] + 2, ram_state[11] + 16
    elif type(player_missile) == PlayerMissile:
        objects[1] = NoObject()

    # ENEMIES
    # The 7 rightmost bits of the ram positions 38 to 44 represent a bitmap of the enemies.
    # Each bit is 1 if there is an enemy in its position and 0 if there is not.
    # RAM 37 is enemy direction 0: ->, 128: <-
    for i in range(6):
        row = format(ram_state[38 + i] & 0x7F, '07b')  # gets a string of the 7 relevant bits
        row = [int(x) for x in row]
        row_y = 19 + i * 12
        for j in range(len(row)):  # max 7
            if row[j] == 1 and ram_state[53] >= i:
                enemy_ship = EnemyShip()
                enemy_ship.y = row_y
                enemy_ship.x = 19 + np.floor((ram_state[36])  * 0.5) + np.ceil(j * 16.5)
            else:
                enemy_ship = NoObject()
            objects[2+ i + 6 * j] = enemy_ship


    # DIVING ENEMIES
    for i in range(5):
        x_pos = ram_state[64 + i] + 7.5
        y_pos = np.ceil(ram_state[69 + i] * 0.74) + 8
        if 8 < y_pos < 186:  # the diving enemy is in the visible area
            diving_enemy = DivingEnemy()
            diving_enemy.x = x_pos
            diving_enemy.y = y_pos
            objects[50 + i] = diving_enemy
        else:
            objects[50 + i] = NoObject()

    # ENEMY MISSILES
    # RAM 48 doppel?
    """
    The missiles are setting ram_state[25:32] as they descend, so these are used for the y calculation. 
    The idea here is that the missiles always use the first unused ram of ram_state[102 + i]
    """

    for i in range(ram_state[53], 13):
        bits = format(prev_ram[i + 20], '#05b')[2:5]
        for j in range(3):
            idx = j + 3 * i
            if bits[j] == '1':
                if type(enemy_missiles[idx]) == NoObject:
                    enemy_missiles[idx] = EnemyMissile()
                enemy_missiles[idx].y = 36 + 12 * i + 4 * j
            else:
                enemy_missiles[idx] = NoObject()

    k = 0
    for i in range(MAX_NB_OBJECTS["EnemyMissile"]):
        if type(enemy_missiles[i]) == EnemyMissile:
            x_ram = ram_state[102 + k % 8]
            if enemy_missiles[i].y >= 184 or x_ram < 5:
                enemy_missiles[i] = NoObject()
            else:
                enemy_missiles[i].x = x_ram + 11
            k += 1
        objects[55 + i] = enemy_missiles[i]

    if hud:
        # Lives
        for i in range(3):
            if ram_state[57] >= i:
                if type(objects[i - 5]) == NoObject:
                    objects[i - 5] = Life()
                    objects[i - 5].x += 5 * i
            else:
                objects[i - 5] = NoObject()
        # Score
        objects[-2].value = _convert_number(ram_state[44]) * 10000 + _convert_number(ram_state[45]) * 100 + _convert_number(ram_state[46])
        # Round
        objects[-1].value = ram_state[47]

    prev_ram = ram_state

def _detect_objects_galaxian_raw(info, ram_state):
    info["score"] = _convert_number(ram_state[44]) * 10000 + _convert_number(ram_state[45]) * 100 + _convert_number(ram_state[46])
    info["lives"] = ram_state[57]
    info["round"] = ram_state[47]


"""
Other Information:

ram 12-17 is another bitmap of the rows of enemies, but this time encoded differently:
    if the bits in a ram cell are numbered like this: 8 7 6 5 4 3 2 1 
    then a row of enemies is represented like this: 3 4 1 2 6 7 5 
    example: if a ram cell holds 01111011, the corresponding row of enemies looks like this: ○ ● ● ● ● ● ● (the leftmost one is missing)
    The benefit/use case of this representation in contrast to the other enemy bitmap is not clear at this point

ram 19-24 changes the pose of the enemy rows (the shorter or taller one and sometimes moved by one pixel along the x-axis)

ram 74-77 changes the direction of the diving enemies
ram 79-82 changes direction and color of diving enemies (so most likely the skin)
ram 86 seems to be related to the bitmap of the diving enemies, which is used when multiple enemies fall together and only use one x and y ram (often happens with one white and two red ones at the end of level 1 if only very few are left)
ram 94 affects the "fall rate" of diving enemies

88, 89 falling

"""