import numpy as np

from .config import ActionSpaceType, LearningType


def handle_player_collision(p1, p2):
    p1.velocity += p2.velocity
    p1.velocity *= 0.5
    p2.velocity = p1.velocity.copy()


def handle_kick(p, b):
    b_movement = p.velocity * 1.25
    p.velocity += b.velocity * 0.25
    return b_movement


def clarify_action(action, movement_vectors, action_type, learning_type, num_players):
    if action_type == ActionSpaceType.CONTINUOUS and learning_type == LearningType.MULTI_AGENT:
        all_action_vectors = action
    elif action_type == ActionSpaceType.CONTINUOUS and learning_type == LearningType.SINGLE_AGENT:
        all_action_vectors = np.zeros((num_players, 2))
        all_action_vectors[0] = movement_vectors
    elif action_type == ActionSpaceType.DISCREETE and learning_type == LearningType.MULTI_AGENT:
        all_action_vectors = movement_vectors[action]
    elif action_type == ActionSpaceType.DISCREETE and learning_type == LearningType.SINGLE_AGENT:
        all_action_vectors = np.zeros((num_players, 2))
        all_action_vectors[0] = movement_vectors[action]
    else:
        assert False
    return all_action_vectors
